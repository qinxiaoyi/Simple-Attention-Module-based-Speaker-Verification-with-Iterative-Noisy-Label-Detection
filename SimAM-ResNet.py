import torch, torch.nn as nn, numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import init

class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1,id_bolck=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        in_dim = int(80/np.power(2,id_bolck-1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out
    
    def SimAM(self,X,lambda_p=1e-4):
        n = X.shape[2] * X.shape[3]-1
        d = (X-X.mean(dim=[2,3], keepdim=True)).pow(2)
        v = d.sum(dim=[2,3], keepdim=True)/n
        E_inv = d / (4*(v+lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)

def ResNet34SimAM(in_planes, **kwargs):
    return ResNet(in_planes, SimAMBasicBlock, [3,4,6,3], **kwargs)      
   
class ResNet34SimAMAtt(nn.Module):
    def __init__(self, in_planes, embedding_size, dropout=0.5,**kwargs):
        super(ResNet34SimAMAtt, self).__init__()
        print('Encoder is ASP')
        self.front = ResNet34SimAM(in_planes, **kwargs)
        outmap_size = int(80/8)
        self.attention = nn.Sequential(
                        nn.Conv1d(in_planes*8 * outmap_size, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Conv1d(128, in_planes*8 * outmap_size, kernel_size=1),
                        nn.Softmax(dim=2),
                        )

        out_dim = in_planes*8 * outmap_size * 2
            
        self.bottleneck = nn.Linear(out_dim, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = x.reshape(x.size()[0],-1,x.size()[-1])
        w = self.attention(x)
        
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat((mu,sg),1)
        
        x = x.view(x.size()[0], -1)
    
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out    

class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, num_classes=10, in_ch=1, feat_dim='2d', **kwargs):
        super(ResNet, self).__init__()
        if feat_dim=='1d':
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim=='2d':
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim=='3d':
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            print('error')

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
