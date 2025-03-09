import data_augmentation
from torchvision import models
import torch
import torch.nn as nn
from timm.models import vision_transformer
from models import ConvBNRelu, ImgEmbed
import attenuations
import einops


class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks, num_bits, channels):
        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        x = self.linear(x)  # b d
        return x


class pre_Swin(nn.Module):
    def __init__(self, num_bits=32):
        super(pre_Swin, self).__init__()
        net = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        # net = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        # net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        net.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=num_bits))
        self.model = net

    def forward(self, x):
        x = self.model(x)
        return x


class pre_Vit(nn.Module):
    def __init__(self, num_bits=32):
        super(pre_Vit, self).__init__()
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        net.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=num_bits))
        self.model = net

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_bits = 48
    imgs = torch.rand(size=[16, 3, 224, 224])
    msgs = torch.rand(size=[16, num_bits])

    # vit_encoder = HiddenEncoder(num_bits=num_bits, num_blocks=4, channels=64)
    # vit_decoder = pre_Swin(num_bits=num_bits)
    vit_decoder = HiddenDecoder(num_blocks=8, num_bits=num_bits, channels=64)

    print(vit_decoder)
    result = vit_decoder(imgs)
    print(result.shape)
