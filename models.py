import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=4, kernel_window_ssvep=59, kernel_window=19, conv_3_dilation=4):
        super().__init__()

        filters = [filters_n1, filters_n1 * 2]

        self.conv_1 = ConvBlock(in_ch=1, out_ch=filters[0], kernel_size=(1, kernel_window_ssvep) ,padding=(0,(kernel_window_ssvep-1)//2))
        self.conv_2 = ConvBlock(in_ch=filters[0], out_ch=filters[0], kernel_size=(num_channel, 1))
        self.conv_3_1 = ConvBlock(in_ch=filters[0], out_ch=filters[1], kernel_size=(1, kernel_window), padding=(0,(kernel_window-1)*(conv_3_dilation-2)), dilation=(1,conv_3_dilation))
        self.conv_3_2 = ConvBlock(in_ch=filters[1], out_ch=filters[1], kernel_size=(1, kernel_window), padding=(0,(kernel_window-1)*(conv_3_dilation-2)), dilation=(1,conv_3_dilation))

        self.pool = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(signal_length*filters[1]//2,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,num_classes)

        self.residual_1 = ResidualBlock(in_ch=filters[0], out_ch=filters[0], kernel_1=kernel_window, kernel_2=(kernel_window+kernel_window_ssvep)//2, kernel_3=kernel_window_ssvep)
        self.residual_2 = ResidualBlock(in_ch=filters[1], out_ch=filters[1], kernel_1=kernel_window, kernel_2=(kernel_window+kernel_window_ssvep)//2, kernel_3=kernel_window_ssvep)

    def forward(self, x):
        x = torch.unsqueeze(x,1)

        x = self.conv_1(x)
        x = self.residual_1(x)
        x = self.dropout(x)

        x = self.conv_2(x)
        x = self.dropout(x)

        x = self.conv_3_1(x)
        x = self.residual_2(x)
        x = self.pool(x)

        x = self.conv_3_2(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=(0,0), dilation=(1,1), w_in=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )
        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_1, kernel_2, kernel_3):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.inception = InceptionBlock(in_ch=in_ch, out_ch=out_ch, kernel_1= kernel_1, kernel_2= kernel_2, kernel_3 =  kernel_3)

    def forward(self, x):
        residual = x
        out = self.inception(x)
        out += residual
        out = self.relu(out)

        return out


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_1, kernel_2, kernel_3):
        super(InceptionBlock,self).__init__()
        out = int(out_ch//4)
        self.kernel_2 = kernel_2
        self.branch1_1 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))

        self.branch2_1 = nn.Conv2d(in_ch,in_ch,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
        self.branch2_2 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_3),padding=(0,(kernel_3-1)//2))

        self.branch3_1 = nn.Conv2d(in_ch,in_ch,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))
        self.branch3_2 = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_2),padding=(0,(kernel_2-1)//2))
        self.branch3_3 = nn.Conv2d(out,out,kernel_size=(1,kernel_2),padding=(0,(kernel_2-1)//2))

        self.branch_pool = nn.Conv2d(in_ch,out,kernel_size=(1,kernel_1),padding=(0,(kernel_1-1)//2))

    def forward(self,x):
        branch1_1 = self.branch1_1(x)

        branch2_1 = self.branch2_1(x)
        branch2_2 = self.branch2_2(branch2_1)

        branch3_1 = self.branch3_1(x)
        branch3_2 = self.branch3_2(branch3_1)
        branch3_3 = self.branch3_3(branch3_2)

        branch_pool = F.avg_pool2d(x, kernel_size=(1, self.kernel_2), stride=1, padding=(0, (self.kernel_2-1)//2))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch_pool, branch1_1, branch2_2, branch3_3]
        return torch.cat(outputs,dim=1)
