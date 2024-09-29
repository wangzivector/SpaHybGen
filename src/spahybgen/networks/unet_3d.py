# Inherent from [3D-UNet](https://github.com/AghdamAmir/3D-UNet)
# Paper URL: https://arxiv.org/abs/1606.06650

# model = UNet3D(in_channels=3, num_classes=1)
# summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")

from torch import nn
import torch
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        

class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels=1, augment=False, voxel_discreteness=80, orientation='quat') -> None: # [64, 128, 256] 512
        super(UNet3D, self).__init__()
        self.orientation = orientation
        level_channels, bottleneck_channel = [16, 32, 64], 128
        if augment: level_channels, bottleneck_channel = [32, 64, 128], 256
        self.augment_heads = augment
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)

        if self.augment_heads:
            self.conv_quat_a = nn.Conv3d(in_channels=level_1_chnls, out_channels=level_1_chnls//2, kernel_size=(3,3,3), padding=(1,1,1))
            self.acti_quat = nn.Tanh()
            self.conv_so3_a = nn.Conv3d(in_channels=level_1_chnls, out_channels=level_1_chnls//2, kernel_size=(3,3,3), padding=(1,1,1))
            self.acti_so3 = nn.Tanh()
            self.conv_wren_a = nn.Conv3d(in_channels=level_1_chnls, out_channels=level_1_chnls//2, kernel_size=(3,3,3), padding=(1,1,1))
            self.acti_wren = nn.Tanh()

        self.conv_score = nn.Conv3d(in_channels=level_1_chnls//(2 if self.augment_heads else 1), out_channels=1, kernel_size=(1,1,1))
        self.conv_wren = nn.Conv3d(in_channels=level_1_chnls//(2 if self.augment_heads else 1), out_channels=1, kernel_size=(1,1,1))

        if self.orientation == "quat": rot_head_size = 4
        elif self.orientation == "so3": rot_head_size = 3
        elif self.orientation == "R6d": rot_head_size = 6
        self.conv_rot = nn.Conv3d(in_channels=level_1_chnls//(2 if self.augment_heads else 1), out_channels=rot_head_size, kernel_size=(1,1,1))
    

    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)

        # predict heads
        out_qua_a = self.acti_quat(self.conv_quat_a(out)) if self.augment_heads else out
        out_score = torch.sigmoid(self.conv_score(out_qua_a))
        out_rot_a = self.acti_so3(self.conv_so3_a(out)) if self.augment_heads else out
        out_rot = F.normalize(self.conv_rot(out_rot_a), dim=1) if self.orientation == "quat" else self.conv_rot(out_rot_a)
        out_wren_a = self.acti_wren(self.conv_wren_a(out)) if self.augment_heads else out
        out_wren = torch.sigmoid(self.conv_wren(out_wren_a))
        return out_score, out_rot, out_wren


# if __name__ == '__main__':
#     model = UNet3D(in_channels=1)
#     from torchsummary import summary
#     summary(model=model, input_size=(1, 80, 80, 80), device='cpu')