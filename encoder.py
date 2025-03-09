import data_augmentation
from torchvision import models
import torch
import torch.nn as nn
from timm.models import vision_transformer
# from models_1 import ConvBNRelu, ImgEmbed
import attenuations
import einops


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class ImgEmbed(nn.Module):
    """ ViT Patch to Image Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, num_patches_w, num_patches_h):
        B, S, CKK = x.shape  # ckk = embed_dim
        x = self.proj(x.transpose(1, 2).reshape(B, CKK, num_patches_h,
                                                num_patches_w))  # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x


class Swin_ImgEmbed(nn.Module):
    """ Swin Transformer Patch to Image Embedding
    """

    def __init__(self, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, S, CKK = x.shape  # ckk = embed_dim
        x = self.proj(x)  # b s (c k k) -> b (c k k) s -> b (c k k) sh sw -> b c h w
        return x


class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks, num_bits, channels, last_tanh=True, input_channel=1):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(input_channel, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + input_channel + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, input_channel, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        msgs = msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1))  # b l h w

        encoded_image = self.conv_bns(imgs)  # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)  # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w


class HiddenEncoder_res(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder_res, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        msgs = msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1))  # b l h w

        encoded_image = self.conv_bns(imgs)  # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)  # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w


class MSA_cross(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image using cross attention.
    """

    def __init__(self, num_bits, img_size=224, patch_size=16, embed_dim=768, last_tanh=True, depth=6, **kwargs):
        super(MSA_cross, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.cross_atten_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.cross_atten_msgs = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        # self.msg_linear = nn.Linear(int(img_size * img_size / (patch_size * patch_size + 1)),
        #                             int(img_size * img_size / (patch_size * patch_size)))
        self.msg_linear = nn.Linear(197, 196)

        self.unpatch = ImgEmbed(embed_dim=embed_dim, patch_size=16)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

        self.depth = depth

    def forward(self, x, msgs):
        num_patches = int(self.patch_embed.num_patches ** 0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x[:, 1:, :]  # without cls token

        msgs = msgs.unsqueeze(1)  # b 1 k
        msgs = msgs.repeat(1, 1, int(x.shape[-1] / msgs.shape[-1]))  # b 1 k -> b 1 (cpq)
        # msgs = einops.rearrange(msgs, 'b l k -> b k l')  # b (cpq) k -> b k (cpq)

        # for ii, blk in enumerate(self.blocks):
        #     x, _ = self.cross_atten(x, msgs, msgs)  # b l (cpq)
        #     x = blk(x)

        for i in range(self.depth):
            x_res, _ = self.cross_atten_img(x, msgs, msgs)
            msgs_res, _ = self.cross_atten_msgs(msgs, x, x)

            x = x + x_res
            msgs = msgs + msgs_res

        img_w = torch.cat((x, msgs), dim=1)
        img_w = einops.rearrange(img_w, 'b l k -> b k l')  # b k+1 (cpq) -> b (cpq) k+1
        img_w = self.msg_linear(img_w)
        img_w = einops.rearrange(img_w, 'b l k -> b k l')  # b (cpq) k -> b k (cpq)

        img_w = self.unpatch(img_w, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class VitEncoder_linear(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_bits, last_tanh=True, patch_size=16, **kwargs):
        super(VitEncoder_linear, self).__init__(**kwargs)
        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.msg_linear = nn.Linear(self.embed_dim + num_bits, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=patch_size)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):
        num_patches = int(self.patch_embed.num_patches ** 0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1)  # b 1 k
        msgs = msgs.repeat(1, x.shape[1], 1)  # b 1 k -> b l k
        for ii, blk in enumerate(self.blocks):
            x = torch.concat([x, msgs], dim=-1)  # b l (cpq+k)
            x = self.msg_linear(x)
            # print(ii, '\n', blk)
            x = blk(x)

        x = x[:, 1:, :]  # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class VitEncoder_cross(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_bits, last_tanh=True, **kwargs):
        super(VitEncoder_cross, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.cross_atten = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        self.msg_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=16)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):

        num_patches = int(self.patch_embed.num_patches ** 0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1)  # b 1 k
        msgs = msgs.repeat(1, x.shape[2], 1)  # b 1 k -> b (cpq) k
        msgs = einops.rearrange(msgs, 'b l k -> b k l')  # b (cpq) k -> b k (cpq)

        for ii, blk in enumerate(self.blocks):
            x, _ = self.cross_atten(x, msgs, msgs)  # b l (cpq)
            x = blk(x)

        x = x[:, 1:, :]  # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class VitEncoder_cross_linear(vision_transformer.VisionTransformer):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_bits, last_tanh=True, **kwargs):
        super(VitEncoder_cross_linear, self).__init__(**kwargs)

        self.head = nn.Identity()
        self.norm = nn.Identity()

        self.cross = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        self.msg_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.unpatch = ImgEmbed(embed_dim=self.embed_dim, patch_size=16)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):

        num_patches = int(self.patch_embed.num_patches ** 0.5)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        msgs = msgs.unsqueeze(1)  # b 1 k

        # watermark for cross attention
        msgs_a = msgs.repeat(1, x.shape[2], 1)  # b 1 k -> b (cpq) k
        msgs_a = einops.rearrange(msgs_a, 'b l k -> b k l')  # b (cpq) k -> b k (cpq)

        # watermark for linear merging
        msgs_l = msgs.repeat(1, x.shape[1], 1)  # b 1 k -> b l k

        for ii, blk in enumerate(self.blocks):
            x, _ = self.cross(x, msgs, msgs)  # b l (cpq)
            x = blk(x)
            x = torch.concat([x, msgs_l], dim=-1)  # b l (cpq+k)
            x = self.msg_linear(x)

        x = x[:, 1:, :]  # without cls token
        img_w = self.unpatch(x, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class pre_Swin_Linear(nn.Module):
    def __init__(self, embed_dim=768, num_bits=32, end_addition=True, last_tanh=True):
        super(pre_Swin_Linear, self).__init__()
        net = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        self.blocks = net.features
        net.head = nn.Sequential(nn.Linear(in_features=768, out_features=num_bits))

        self.norm = net.norm
        self.avgpool = net.avgpool

        self.msg_linear = nn.Linear(embed_dim + 4 * num_bits, embed_dim)
        self.Img_emb = Swin_ImgEmbed(patch_size=32, in_chans=3, embed_dim=embed_dim)

        self.end_addition = end_addition
        self.end_addition_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):
        msgs = msgs.unsqueeze(-2).unsqueeze(-2)

        # for ii, blk in enumerate(self.blocks):
        #     # if ii > 0:
        #     #     msgs = msgs.repeat(1, x.shape[1], x.shape[2], ii)  # b 1 k -> b l k
        #     #     x = torch.concat([x, msgs], dim=-1)  # b l (cpq+k)
        #     #     x = self.msg_linear(x)
        #     x = blk(x)
        #     # print('222222222222\n', ii, '\n', blk, '\n222222222222222222222')
        x = self.blocks(x)
        msgs = msgs.repeat(1, x.shape[1], x.shape[2], 4)  # b 1 1 k -> b l l k
        img_w = torch.concat([x, msgs], dim=-1)  # b l (cpq+k)
        img_w = self.msg_linear(img_w)
        img_w = self.norm(img_w)  # -> b 7 7 768
        img_w = img_w.permute(0, 3, 1, 2)  # -> b 768 7 7

        # whether embed original tokens into watermarked token in the end
        if self.end_addition:
            img_w = img_w.permute(0, 2, 3, 1)  # b 768 7 7 -> b 7 7 768
            img_w = torch.cat([img_w, x], dim=3)  # b 768 7 7 -> b 7 7 768*2
            img_w = self.end_addition_layer(img_w)  # b 7 7 768*2 -> b 7 7 768
            img_w = img_w.permute(0, 3, 1, 2)  # -> b 768 7 7

        img_w = self.Img_emb(img_w)
        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class pre_Swin_cross_atten(nn.Module):
    def __init__(self, embed_dim=768, num_bits=32, end_addition=True, last_tanh=True):
        super(pre_Swin_cross_atten, self).__init__()
        net = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        self.blocks = net.features
        net.head = nn.Sequential(nn.Linear(in_features=768, out_features=num_bits))
        self.num_bits = num_bits

        self.norm = net.norm
        self.avgpool = net.avgpool

        self.msg_cross_atten = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.Img_emb = Swin_ImgEmbed(patch_size=32, in_chans=3, embed_dim=embed_dim)

        self.end_addition = end_addition
        self.end_cross_atten = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):
        msgs = msgs.unsqueeze(-2).unsqueeze(-2)

        x = self.blocks(x)
        msgs = msgs.repeat(1, x.shape[1], x.shape[2], int(x.shape[-1] / self.num_bits))  # b 1 1 k -> b l l k
        msgs = msgs.view(msgs.shape[0], msgs.shape[1] * msgs.shape[2], msgs.shape[-1])
        x_trans = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[-1])
        # x_trans = einops.rearrange(x, 'b h w c -> b (h w) c')
        img_w, _ = self.msg_cross_atten(x_trans, msgs, msgs)  # b l (cpq)

        # whether embed original tokens into watermarked token in the end
        if self.end_addition:
            img_w, _ = self.end_cross_atten(x_trans, img_w, img_w)

        img_w = img_w.view(x.shape[0], x.shape[1], x.shape[2], x.shape[-1])
        img_w = img_w.permute(0, 3, 1, 2)  # -> b 768 7 7

        img_w = self.Img_emb(img_w)

        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


class pre_ViT_Linear(nn.Module):
    def __init__(self, embed_dim=768, num_bits=32, last_tanh=True, patch_size=16, end_addition=True):
        super(pre_ViT_Linear, self).__init__()
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        self._process_input = net._process_input
        self.class_token = net.class_token
        self.encoder = net.encoder
        self.unpatch = ImgEmbed(embed_dim=embed_dim, patch_size=patch_size)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()
        self.end_addition = end_addition

        self.msg_linear = nn.Linear(196 + num_bits, 196)
        self.end_addition_layer = nn.Linear(2 * 196, 196)

    def forward(self, x: torch.Tensor, msgs: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 1:, :]

        # embed watermark into imgs
        msgs = msgs.unsqueeze(-1)
        msgs = msgs.repeat(1, 1, x.shape[2])  # b 1 k -> b l k

        img_w = torch.cat([x, msgs], dim=1)
        img_w = img_w.transpose(1, 2)
        img_w = self.msg_linear(img_w)
        img_w = img_w.transpose(1, 2)

        # whether embed original tokens into watermarked token in the end
        if self.end_addition:
            img_w = torch.cat([img_w, x], dim=1)
            img_w = img_w.transpose(1, 2)
            img_w = self.end_addition_layer(img_w)
            img_w = img_w.transpose(1, 2)

        num_patches = int(x.shape[1] ** 0.5)
        img_w = self.unpatch(img_w, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)
        return img_w


class pre_ViT_cross_attn(nn.Module):
    def __init__(self, embed_dim=768, num_bits=32, last_tanh=True, patch_size=16, end_addition=True):
        super(pre_ViT_cross_attn, self).__init__()
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        self._process_input = net._process_input
        self.class_token = net.class_token
        self.encoder = net.encoder
        self.unpatch = ImgEmbed(embed_dim=embed_dim, patch_size=patch_size)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()
        self.end_addition = end_addition

        # self.msg_linear = nn.Linear(196 + num_bits, 196)
        self.msg_cross_atten = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.end_cross_atten = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=num_bits))

    def forward(self, x: torch.Tensor, msgs: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 1:, :]

        # embed watermark into imgs
        msgs = msgs.unsqueeze(-1)
        msgs = msgs.repeat(1, 1, x.shape[2])  # b 1 k -> b l k

        img_w, _ = self.msg_cross_atten(x, msgs, msgs)  # b l (cpq)

        # whether embed original tokens into watermarked token in the end
        if self.end_addition:
            img_w, _ = self.end_cross_atten(x, img_w, img_w)  # b l (cpq)

        num_patches = int(x.shape[1] ** 0.5)
        img_w = self.unpatch(img_w, num_patches, num_patches)

        if self.last_tanh:
            img_w = self.tanh(img_w)
        return img_w


class pre_Swin_Linear_test(nn.Module):
    def __init__(self, embed_dim=768, num_bits=32, end_addition=True, last_tanh=True):
        super(pre_Swin_Linear_test, self).__init__()
        net = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        self.blocks = net.features
        net.head = nn.Sequential(nn.Linear(in_features=768, out_features=num_bits))

        self.norm = net.norm
        self.avgpool = net.avgpool

        self.msg_linear = nn.Linear(embed_dim + 4 * num_bits, embed_dim)
        self.Img_emb = Swin_ImgEmbed(patch_size=32, in_chans=3, embed_dim=embed_dim)

        self.end_addition = end_addition
        self.end_addition_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x, msgs):
        msgs = msgs.unsqueeze(-2).unsqueeze(-2)

        for ii, blk in enumerate(self.blocks):
            # if ii > 0:
            #     msgs = msgs.repeat(1, x.shape[1], x.shape[2], ii)  # b 1 k -> b l k
            #     x = torch.concat([x, msgs], dim=-1)  # b l (cpq+k)
            #     x = self.msg_linear(x)
            x = blk(x)
            # print('222222222222\n', ii, '\n', blk, '\n222222222222222222222')
        # x = self.blocks(x)
        msgs = msgs.repeat(1, x.shape[1], x.shape[2], 4)  # b 1 1 k -> b l l k
        img_w = torch.concat([x, msgs], dim=-1)  # b l (cpq+k)
        img_w = self.msg_linear(img_w)
        img_w = self.norm(img_w)  # -> b 7 7 768
        img_w = img_w.permute(0, 3, 1, 2)  # -> b 768 7 7

        # whether embed original tokens into watermarked token in the end
        if self.end_addition:
            img_w = img_w.permute(0, 2, 3, 1)  # b 768 7 7 -> b 7 7 768
            img_w = torch.cat([img_w, x], dim=3)  # b 768 7 7 -> b 7 7 768*2
            img_w = self.end_addition_layer(img_w)  # b 7 7 768*2 -> b 7 7 768
            img_w = img_w.permute(0, 3, 1, 2)  # -> b 768 7 7

        img_w = self.Img_emb(img_w)
        if self.last_tanh:
            img_w = self.tanh(img_w)

        return img_w


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_bits = 48
    imgs = torch.rand(size=[16, 3, 224, 224])
    msgs = torch.rand(size=[16, num_bits])

    vit_encoder = HiddenEncoder(num_bits=num_bits, num_blocks=4, channels=64, input_channel=3)
    # vit_encoder = VitEncoder_cross(num_bits=num_bits)
    # vit_encoder = pre_ViT_cross_attn(num_bits=num_bits)
    # vit_encoder = pre_Swin_Linear_test(num_bits=num_bits)
    # vit_encoder = MSA_cross(num_bits=num_bits)

    print(vit_encoder)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    result = vit_encoder(imgs, msgs)
    print(result.shape)
