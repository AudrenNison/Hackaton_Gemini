import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 

class UNET(nn.Module):
    
    def __init__(self, in_channels=3, classes=1):
        super(UNET, self).__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        
        return x 

#####################################################################################################
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, channels, height, width):
        super(TemporalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x_flat = x.view(B, T, -1)  # Flatten spatial dimensions
        attention_weights = self.softmax(torch.bmm(x_flat, x_flat.transpose(1, 2)))
        attention_output = torch.bmm(attention_weights, x_flat).view(B, T, C, H, W)
        return attention_output

class UNetWithAttention_HSI_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention_HSI_3D, self).__init__()
        self.enc1 = ConvBlock3D(in_channels, 64)
        self.enc2 = ConvBlock3D(64, 128)
        self.enc3 = ConvBlock3D(128, 256)
        self.enc4 = ConvBlock3D(256, 512)

        self.pool = nn.MaxPool3d(2)
        
        self.attention = TemporalAttention(512, 32, 32)  # Temporal attention block remains 2D
        
        self.upconv1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.dec1 = ConvBlock3D(512, 256)
        self.dec2 = ConvBlock3D(256, 128)
        self.dec3 = ConvBlock3D(128, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Temporal attention
        e4 = self.attention(e4.unsqueeze(1))  # Assuming single time-step for demo

        # Decoding + Skip connections
        d1 = self.upconv1(e4.squeeze(1))
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.final_conv(d3)
        return out

