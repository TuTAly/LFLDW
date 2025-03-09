# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchmetrics import StructuralSimilarityIndexMeasure

import argparse
import datetime
import json
import os
import time
import random
import ssl

import torchvision.transforms.functional
from PIL import Image
import numpy as np
from pathlib import Path
import warnings

import torch_dct
from torchvision.transforms import functional
from torchvision.utils import save_image
import encoder_unet
import data_augmentation

import inversion_ddim_demo as DDIM

import utils
import utils_img

import models
import attenuations
import encoder_DDIM
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from torchvision import transforms

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 定义 combined_attack 函数
def combined_attack(x, attacks_list, num_attacks=None):
    """
    对输入图像 x 应用多个攻击。
    :param x: 输入图像
    :param attacks_list: 所有可用攻击的字典
    :param num_attacks: 随机选择的攻击数量，None 表示应用所有攻击
    """
    attacked_x = x.clone()  # 确保不修改原始输入
    attack_funcs = list(attacks_list.values())  # 提取所有攻击函数
    attacks_to_apply = attack_funcs if num_attacks is None else random.sample(attack_funcs, num_attacks)
    for attack in attacks_to_apply:
        attacked_x = attack(attacked_x)  # 顺序应用攻击
    return attacked_x


def sample_from_tensors(filename='stage1_35/ori_latent', sample_num=18, seed=None):
    if seed == None:
        pass
    else:
        random.seed(seed)

    # 定义保存张量文件的文件夹路径
    tensor_folder_path = filename

    # 获取文件夹中所有的张量文件名
    tensor_files = [f for f in os.listdir(tensor_folder_path) if f.endswith('.pt')]

    # 定义要随机选取的张量数量
    x = sample_num  # 替换为您需要的数量

    # 确保 x 不超过文件总数
    x = min(x, len(tensor_files))

    # 随机选取 x 个张量文件
    selected_files = random.sample(tensor_files, x)

    # 初始化一个列表来存储加载的张量
    tensors = []

    # 批量加载选取的张量文件
    for tensor_file in selected_files:
        tensor_path = os.path.join(tensor_folder_path, tensor_file)
        tensor = torch.load(tensor_path)
        tensors.append(tensor)

    # 将选取的张量合并为一个大的四维张量
    # 假设所有张量具有相同的形状
    all_tensors = torch.stack(tensors)

    all_tensors = all_tensors.squeeze()
    return all_tensors


def fft_image(image):
    # 对图像的最后两个维度进行二维傅里叶变换
    # image_fft = torch.fft.fft2(image,norm="forward")
    image_fft = torch.fft.fft2(image)
    image_fft = torch.fft.fftshift(image_fft, dim=(-1, -2))
    # image_fft = torch.abs(image_fft)
    # 返回复数形式的傅里叶变换结果
    return image_fft


def ifft_image(image_fft):
    # 对复数形式的傅里叶变换结果进行逆变换
    image_fft = torch.fft.ifftshift(image_fft, dim=(-1, -2))
    image = torch.fft.ifft2(image_fft).real
    # image = torch.fft.ifft2(image_fft,norm="forward").real
    # 返回逆变换后的实部
    return image


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def create_concentric_circle_mask(size, center, radius, radius_2, num_masked=0.5, num_unmasked=0):
    # 创建一个全零数组作为掩码 1:其他地方为1 2:其他地方为0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = torch.zeros(size)
    center = torch.tensor(center)
    # 生成同心圆形状的掩码
    for y in range(size[0]):
        for x in range(size[1]):
            # calculate the distance between the center of circle and the target points
            distance_to_center = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
            # the area in the concentric circles
            if radius <= distance_to_center < radius_2:
                mask[y, x] = num_masked
            # the area not in the concentric circles
            else:
                mask[y, x] = num_unmasked
    mask = mask.to(device)
    return mask


def add_noise(image, msgs, channel=0, r1=10, w_weight=0.9, w_width=1, w_value=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_1 = torch.unsqueeze(image[:, channel], dim=1).real
    temp_2 = torch.unsqueeze(image[:, channel], dim=1).imag
    temp = torch.cat((temp_1, temp_2), dim=1)
    for ii in range(len(msgs)):
        mean_w = torch.mean(temp_1[ii])
        for i in range(len(msgs[0])):
            masked_area = create_concentric_circle_mask(size=(64, 64), center=(32, 32),
                                                        radius=1 * i + params.r1,
                                                        radius_2=1 * i + params.r1 + 1,
                                                        num_masked=0,
                                                        num_unmasked=1, )

            if msgs[ii, i] == 1:
                w_latent = create_concentric_circle_mask(size=(64, 64), center=(32, 32),
                                                         radius=1 * i + params.r1,
                                                         radius_2=1 * i + params.r1 + 1,
                                                         num_masked=mean_w + w_value,
                                                         num_unmasked=0)
                temp_2 = temp_1[ii] * masked_area
                temp_1[ii] = temp_2 + w_latent

                # a = temp_1[0, 0].cpu().numpy()
                # plt.title('will be deleted 00000000000')
                # plt.imshow(a, cmap='gray')
                # plt.show()

                # init_latents_w_cos_certain_channel = torch.from_numpy(temp_1).to(device)
            else:
                w_latent = create_concentric_circle_mask(size=(64, 64), center=(32, 32),
                                                         radius=1 * i + params.r1,
                                                         radius_2=1 * i + params.r1 + 1,
                                                         num_masked=mean_w - w_value,
                                                         num_unmasked=0)
                # w_latent = torch.from_numpy(w_latent).to(device)
                # init_latents_w_cos_certain_channel = temp + w_latent
                temp_2 = temp_1[ii] * masked_area
                temp_1[ii] = temp_2 + w_latent

                # a = temp_1[0, 0].cpu().numpy()
                # plt.title('will be deleted 11111111')
                # plt.imshow(a, cmap='gray')
                # plt.show()

    a = temp_1[0, 0].cpu().numpy()
    plt.title('will be deleted 11111111')
    plt.imshow(a, cmap='gray')
    plt.show()
    weighted_watermark = (1 - w_weight) * temp + w_weight * temp_1
    complex = torch.complex(weighted_watermark[:, 0], temp[:, 1]).to(device)
    image[:, channel] = complex
    return image


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="../stable_signature/dataset/coco-mini-mini-test-copy")
    # aa("--train_dir", type=str, default="imgs")

    aa("--val_dir", type=str, default="../stable_signature/dataset/coco-mini-mini-test")
    aa("--output_dir", type=str, default="V06.27/", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')
    aa("--r1", type=int, default=5, help="frequency domain lowest limitation")
    aa("--r2", type=int, default=5 + 8, help="frequency domain highest limitation")
    aa("--inversion", type=str, default=None, help="the type of DDIM inversion method")
    # aa("--inversion", type=str, default='edict', help="the type of DDIM inversion method")

    aa("--num_bits", type=int, default=8, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=512, help="Image size")
    aa("--total_step", type=int, default=50, help="Total sample steps of diffusion")
    aa("--start_step", type=int, default=35, help="start inversion step")
    aa("--prompt", type=str, default='Beautiful DSLR Photograph of the beach, golden hour,landscape,sea,blue sea,sand',
       help="The prompt used in diffusion")
    aa("--negative_prompt", type=str,
       default='nsfw, paintings, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry, Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, bad anatomy, amputation, bad proportions, twins, missing body, fused body, extra head, poorly drawn face, bad eyes, deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms, bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands, connected hand, bad hands, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands, extra legs, bad legs, many legs, more than two legs, bad feet, extra feets, badhandv4, easynegative, FastNegativeV2, negative_hand-neg,ng_deepnegative_v1_75t, verybadimagenegative_v1.3',
       help="The negative prompt used in diffusion")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="unet", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=24, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=10, type=int)
    aa('--saveckp_freq', default=100, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=400, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam,lr=2e-2", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")

    # aa("--scheduler", type=str, default='CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5',
    #    help="Scheduler to use. (Default: None)")

    aa("--lambda_w", type=float, default=1, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=20, help="Weight of the image loss. (Default: 0.0)")
    aa("--loss_margin", type=float, default=1,
       help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='l1',
       help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce',
       help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=2, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=4, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=1, help="Number of workers for data loading. (Default: 8)")
    aa("--w_weight", type=float, default=1.0, help="alpha*img+beta*watermark and beta is the w_weight")

    group = parser.add_argument_group('Attenuation parameters')
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="combined",
       help="Type of data augmentation to use at marking time. (Default: combined)")
    # aa("--p_crop", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    # aa("--p_res", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    # aa("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    # aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    # aa("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    # aa("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")

    aa("--p_crop", type=float, default=1, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_res", type=float, default=0, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_blur", type=float, default=1, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_jpeg", type=float, default=1, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    aa("--p_rot", type=float, default=1, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_noise", type=float, default=1, help="Probability of the gaussian noise augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=1, help="Probability of the color jitter augmentation. (Default: 0.5)")

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=False, help='Enabling distributed training')

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')
    aa('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    # aa('--dataset', default='coco')
    # aa('--dataset', default='cifar100')
    aa('--num_images', default=1, type=int)
    # aa('--guidance_scale', default=1, type=float)
    aa('--guidance_scale', default=7.5, type=float)
    aa('--num_inference_steps', default=50, type=int)
    aa('--image_length', default=512, type=int)

    return parser


def main(params):
    global device
    ssl._create_default_https_context = ssl._create_unverified_context
    # Distributed mode
    if params.dist:
        utils.init_distributed_mode(params)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    # Set seeds for reproductibility
    seed = params.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # handle params that are "none"
    if params.attenuation is not None:
        if params.attenuation.lower() == 'none':
            params.attenuation = None
    if params.scheduler is not None:
        if params.scheduler.lower() == 'none':
            params.scheduler = None

    # Build encoder
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = encoder_DDIM.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits,
                                             channels=params.encoder_channels, last_tanh=params.use_tanh,
                                             input_channel=1)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits,
                                       channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'vit':
        encoder = models.VitEncoder(
            img_size=params.img_size, patch_size=16, init_values=None,
            embed_dim=params.encoder_channels, depth=params.encoder_depth,
            num_bits=params.num_bits, last_tanh=params.use_tanh
        )
    elif params.encoder == 'unet':
        encoder = encoder_unet.UNet(
            n_channels=1, n_classes=1, num_bits=params.num_bits, bilinear=True)
    else:
        raise ValueError('Unknown encoder type')
    encoder = encoder.to(device)
    print('\nencoder: \n%s' % encoder)
    print('total parameters: %d' % sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits * params.redundancy,
                                       channels=params.decoder_channels, input_channel=1)
    elif params.decoder == 'pre_swin':
        decoder = models.pre_Swin(num_bits=params.num_bits * params.redundancy)
    else:
        raise ValueError('Unknown decoder type')
    decoder = decoder.to(device)
    print('\ndecoder: \n%s' % decoder)
    print('total parameters: %d' % sum(p.numel() for p in decoder.parameters()))

    # Adapt bn momentum
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None

    # Construct attenuation
    if params.attenuation == 'jnd':
        attenuation = attenuations.JND(preprocess=lambda x: utils_img.unnormalize_rgb(x)).to(device)
    else:
        attenuation = None

    # Construct data augmentation seen at train time
    if params.data_augmentation == 'combined':
        data_aug = data_augmentation.HiddenAug(params.img_size, params.p_crop, params.p_blur, params.p_jpeg,
                                               params.p_rot, params.p_color_jitter, params.p_res, params.p_noise).to(
            device)
    elif params.data_augmentation == 'kornia':
        data_aug = data_augmentation.KorniaAug().to(device)
    elif params.data_augmentation == 'none':
        data_aug = nn.Identity().to(device)
    else:
        raise ValueError('Unknown data augmentation type')
    print('data augmentation: %s' % data_aug)

    # Create encoder/decoder
    encoder_decoder = models.EncoderDecoder(encoder, attenuation, data_aug, decoder,
                                            params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits,
                                            params.redundancy)

    encoder_decoder = encoder_decoder.to(device)

    # Distributed training
    if params.dist:
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
        encoder_decoder = nn.parallel.DistributedDataParallel(encoder_decoder, device_ids=[params.local_rank])

    # Build optimizer and scheduler
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(
        params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # Data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.img_size),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # maybe we don't need it
        # utils_img.normalize_rgb,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # utils_img.normalize_rgb,
    ])
    train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size,
                                        num_workers=params.workers, shuffle=True)
    val_loader = utils.get_dataloader(params.val_dir, transform=val_transform, batch_size=params.batch_size_eval,
                                      num_workers=params.workers, shuffle=False)

    # optionally resume training
    if params.resume_from is not None:
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder_decoder=encoder_decoder,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    print('training...')
    start_time = time.time()
    best_bit_acc = 0

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_id = 'stabilityai/stable-diffusion-2-1-base'
    # model_id='runwayml/stable-diffusion-v1-5'
    model_id = 'stabilityai/stable-diffusion-2-base'

    dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')

    pipeline = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=dpm_scheduler,
        torch_dtype=torch.float32,
        revision='fp16',
    )

    pipeline = pipeline.to(device)

    for epoch in range(start_epoch, params.epochs):
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        if epoch % params.eval_freq == 0:
            val_stats = eval_one_epoch(encoder_decoder, val_loader, epoch, params, pipe=pipeline)
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}

        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        save_dict_encoder = {
            'encoder': encoder_decoder.encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        save_dict_decoder = {
            'encoder': encoder_decoder.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        utils.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        utils.save_on_master(save_dict_encoder, os.path.join(params.output_dir, f'checkpoint_encodoer_{epoch:03}.pth'))
        utils.save_on_master(save_dict_decoder, os.path.join(params.output_dir, f'checkpoint_decodoer_{epoch:03}.pth'))
        if params.saveckp_freq and epoch % params.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth'))
            utils.save_on_master(save_dict_encoder,
                                 os.path.join(params.output_dir, f'checkpoint_encodoer_{epoch:03}.pth'))
            utils.save_on_master(save_dict_decoder,
                                 os.path.join(params.output_dir, f'checkpoint_decodoer_{epoch:03}.pth'))
        if utils.is_main_process():
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def message_loss(fts, targets, m, loss_type='mse'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts / m), 0.5 * (targets + 1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')


def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')


@torch.no_grad()
def eval_one_epoch(encoder_decoder: models.EncoderDecoder, loader, epoch, params, pipe, is_mask=True):
    """
    One epoch of eval.
    """
    warnings.filterwarnings("ignore")

    encoder_decoder.eval()
    header = 'Eval - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    total_step = params.total_step
    watermarked_step = params.start_step

    # get text embedding
    dataset, prompt_key = get_dataset(params)
    tester_prompt_temp = ''  # assume at the detection time, the original prompt is unknown
    tester_prompt = []
    for i in range(params.batch_size_eval):
        tester_prompt.append(tester_prompt_temp)
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    for it, (imgs, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        # imgs = imgs.to(device, non_blocking=True)  # b c h w

        init_latents_no_w = pipe.get_random_latents(batch=params.batch_size_eval)

        # current_prompt = dataset[it][prompt_key]
        current_prompt = []
        for i in range(params.batch_size_eval):
            current_prompt.append(dataset[it][prompt_key])

        # change the param stop_step to dertermine the injecting watermark timestep
        stop_step = params.start_step

        # print(current_prompt)

        outputs_latents_no_w = pipe(
            current_prompt,
            num_images_per_prompt=params.num_images,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            height=params.image_length,
            width=params.image_length,
            latents=init_latents_no_w,
            if_stop=True,
            stop_step=stop_step,
        )

        # generate image without wataermark and save them in G
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=params.num_images,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            height=params.image_length,
            width=params.image_length,
            latents=outputs_latents_no_w,
            if_start=True,
            stop_latents=outputs_latents_no_w,
            stop_step=stop_step
        )

        # show and save original image
        orig_image_no_w = outputs_no_w.images[0]
        file_path = f"G_W4/ori_image_epoch_{epoch}iteration_{it}.png"
        orig_image_no_w.save(file_path)
        # plt.title('origianl image')
        # plt.imshow(orig_image_no_w)
        # plt.show()

        # generate random watermark code
        msgs_ori = torch.rand((outputs_latents_no_w.shape[0], params.num_bits)) > 0.5  # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1  # b k

        # transform internel stable diffusion latents to Fourier domain
        latents = fft_image(outputs_latents_no_w).to(device)

        temp1 = torch.unsqueeze(latents[:, 0], dim=1).real
        temp2 = torch.unsqueeze(latents[:, 0], dim=1).imag
        temp = torch.cat((temp1, temp2), dim=1)

        # show and save the latent code of original image
        b = temp[0, 0].cpu().numpy()
        a = temp[0, 1].cpu().numpy()
        plt.title('original image latent code real part')
        # plt.imshow(b, cmap='gray')
        # plt.show()

        # plt.title('original image latent code imaginary part')
        # plt.imshow(a, cmap='gray')
        # plt.show()

        # inject watermark using deep learning network
        # temp is the original latent code and temp_1 is the watermarked latent code
        # temp_1 = temp.cpu().numpy()

        # print('msgs:', msgs)

        complex = add_noise(latents, msgs, r1=params.r1, w_weight=params.w_weight, channel=0)

        inv_latents = ifft_image(complex).clone().to(device)

        # generate image with watermark
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=params.num_images,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            height=params.image_length,
            width=params.image_length,
            latents=inv_latents,
            if_start=True,
            stop_latents=inv_latents,
            stop_step=stop_step
        )

        # show and save watermarked image
        orig_image_w = outputs_w.images[0]
        file_path = f"V_G_W/w_image_epoch_{epoch}iteration_{it}.png"
        orig_image_w.save(file_path)

        orig_image_no_w_tensor = torch.zeros(size=(params.batch_size_eval, 3, 512, 512)).to(device)
        for i in range(params.batch_size_eval):
            orig_image_no_w_tensor[i] = transform_img(outputs_no_w.images[i]).to(text_embeddings.dtype).to(device)

        orig_image_w_tensor = torch.zeros(size=(params.batch_size_eval, 3, 512, 512)).to(device)
        for i in range(params.batch_size_eval):
            orig_image_w_tensor[i] = transform_img(outputs_w.images[i]).to(text_embeddings.dtype).to(device)

        image_latents_w = pipe.get_image_latents(orig_image_w_tensor, sample=False)

        if params.inversion == None:
            inv_latents_w_list = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=params.num_inference_steps,
                if_list=True,
            )
        elif params.inversion == 'edict':
            inv_latents_w_list = pipe.forward_diffusion_2(
                x=image_latents_w,
                y=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=params.num_inference_steps,
                if_list=True,
            )

        inv_latents_w = inv_latents_w_list[-1 - stop_step]
        inverted_latents = fft_image(inv_latents_w)

        temp1 = torch.unsqueeze(inverted_latents[:, 0], dim=1).real
        temp2 = torch.unsqueeze(inverted_latents[:, 0], dim=1).imag
        temp = torch.cat((temp1, temp2), dim=1)

        b = temp[0, 1].cpu().numpy()
        a = temp[0, 0].cpu().numpy()
        # plt.title('inverted latent code real part')
        # plt.imshow(a, cmap='gray')
        # file_path = f"Latent_code_i/inverted_image_epoch_{epoch}iteration_{it}.png"
        # plt.savefig(file_path)
        # plt.show()

        # plt.title('inverted latent code imaginary part')
        # plt.imshow(b, cmap='gray')
        # plt.show()

        # problem is here!!!!!!!!!!

        fts, _ = encoder_decoder(temp1, msgs, r1=params.r1, r2=params.r2,
                                 have_decoder=True, have_encoder=False)

        loss_w = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type)  # b -> 1
        loss_i = image_loss(orig_image_no_w_tensor, orig_image_w_tensor, loss_type=params.loss_i_type)  # b c h w -> 1

        loss = params.lambda_w * loss_w + params.lambda_i * loss_i

        # img stats
        psnrs = utils_img.psnr(orig_image_no_w_tensor, orig_image_w_tensor)  # b 1

        ssim = StructuralSimilarityIndexMeasure()
        ssims = ssim(orig_image_no_w_tensor.cpu(), orig_image_w_tensor.cpu())

        # msg stats
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0  # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs))  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        word_accs = (bit_accs == 1)  # b
        norm = torch.norm(fts, dim=-1, keepdim=True)  # b d -> b 1
        log_stats = {
            'loss_w': loss_w.item(),
            'loss_i': loss_i.item(),
            'loss': loss.item(),
            'psnr_avg': torch.mean(psnrs).item(),
            'ssim_avg': torch.mean(ssims).item(),
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'word_acc_avg': torch.mean(word_accs.type(torch.float)).item(),
            'norm_avg': torch.mean(norm).item(),
        }

        attacks_list = {
            # 'none': lambda x: x,
            # 'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            # 'crop_09': lambda x: utils_img.center_crop(x, 0.9),
            # 'resize_03': lambda x: utils_img.resize(x, 0.3),
            # 'resize_05': lambda x: utils_img.resize(x, 2),
            # 'rot_10': lambda x: utils_img.rotate(x, 10),
            # 'rot_90': lambda x: utils_img.rotate(x, 90),
            # 'blur_size_11': lambda x: utils_img.gaussian_blur(x, sigma=4.0, kernel_size=11),
            # 'blur_size_9': lambda x: utils_img.gaussian_blur(x, sigma=4.0, kernel_size=9),
            # 'blur_size_7': lambda x: utils_img.gaussian_blur(x, sigma=4.0, kernel_size=7),
            # 'blur_size_5': lambda x: utils_img.gaussian_blur(x, sigma=4.0, kernel_size=5),
            # 'blur_size_3': lambda x: utils_img.gaussian_blur(x, sigma=4.0, kernel_size=3),
            # 'blur_1': lambda x: utils_img.gaussian_blur(x, sigma=1, kernel_size=3),
            # 'blur_2': lambda x: utils_img.gaussian_blur(x, sigma=2, kernel_size=3),
            # 'blur_3': lambda x: utils_img.gaussian_blur(x, sigma=3, kernel_size=3),
            # 'blur_4': lambda x: utils_img.gaussian_blur(x, sigma=4, kernel_size=3),
            # 'blur_5': lambda x: utils_img.gaussian_blur(x, sigma=5, kernel_size=3),
            # 'blur_16': lambda x: utils_img.gaussian_blur(x, sigma=16, kernel_size=3),
            # 'blur_64': lambda x: utils_img.gaussian_blur(x, sigma=64, kernel_size=3),
            # 'blur_128': lambda x: utils_img.gaussian_blur(x, sigma=128, kernel_size=3),
            # 'blur_0.5': lambda x: utils_img.gaussian_blur(x, sigma=0.5, kernel_size=3),
            # 'blur_0.8': lambda x: utils_img.gaussian_blur(x, sigma=0.8, kernel_size=3),
            # 'blur_2.5': lambda x: utils_img.gaussian_blur(x, sigma=2.5, kernel_size=3),
            # 'brightness_1.5': lambda x: utils_img.adjust_brightness(x, 1.5),
            # 'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            # 'brightness_2.5': lambda x: utils_img.adjust_brightness(x, 2.5),
            # 'brightness_3': lambda x: utils_img.adjust_brightness(x, 3),
            # 'brightness_3.5': lambda x: utils_img.adjust_brightness(x, 3.5),
            # # 'jpeg_30': lambda x: utils_img.jpeg_compress(x, 30),
            # # 'jpeg_40': lambda x: utils_img.jpeg_compress(x, 40),
            # # 'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            # # 'jpeg_60': lambda x: utils_img.jpeg_compress(x, 60),
            'jpeg_70': lambda x: utils_img.jpeg_compress(x, 70),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'gaussian_noise_0.1': lambda x: utils_img.gaussian_noise(x, std=0.1),
            # 'gaussian_noise_0.3': lambda x: utils_img.gaussian_noise(x, std=0.3),
            # 'gaussian_noise_0.5': lambda x: utils_img.gaussian_noise(x, std=0.5),
            # 'gaussian_noise_0.7': lambda x: utils_img.gaussian_noise(x, std=0.7),
            # 'gaussian_noise_0.9': lambda x: utils_img.gaussian_noise(x, std=0.9),
            # 'blur_2': lambda x: utils_img.gaussian_blur(x, sigma=2, kernel_size=3),
            'resize_05': lambda x: utils_img.resize(x, 0.5),
        }

        attacks = {
            'combined': lambda x: combined_attack(x,attacks_list),
        }

        for name, attack in attacks.items():
            imgs_aug = attack(orig_image_w_tensor)
            if imgs_aug.shape != imgs.shape:
                imgs_aug = F.interpolate(imgs_aug, size=params.img_size, mode='bilinear')

            with torch.no_grad():
                image_latents_w = pipe.get_image_latents(imgs_aug, sample=False)

            if params.inversion == None:
                inv_latents_w_list = pipe.forward_diffusion(
                    latents=image_latents_w,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=params.num_inference_steps,
                    if_list=True,
                )
            elif params.inversion == 'edict':
                inv_latents_w_list = pipe.forward_diffusion_2(
                    x=image_latents_w,
                    y=image_latents_w,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=params.num_inference_steps,
                    if_list=True,
                )

            inv_latents_w = inv_latents_w_list[-1 - stop_step]

            inverted_latents = fft_image(inv_latents_w)

            temp1 = torch.unsqueeze(inverted_latents[:, 0], dim=1).real
            temp2 = torch.unsqueeze(inverted_latents[:, 0], dim=1).imag
            temp = torch.cat((temp1, temp2), dim=1)

            # plt.title('latent code 1')
            # plt.imshow(inv_latents_w_cos_certain_channel[0, 0].cpu(), cmap='gray')
            # plt.show()

            fts, _ = encoder_decoder(temp1, msgs, r1=params.r1, r2=params.r2, have_decoder=True, have_encoder=False)

            decoded_msgs = torch.sign(fts) > 0  # b k -> b k
            diff = (~torch.logical_xor(ori_msgs, decoded_msgs))  # b k -> b k
            log_stats[f'bit_acc_{name}'] = diff.float().mean().item()

            attack_bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
            attack_word_accs = (attack_bit_accs > 0.85)  # b

            log_stats[f'word_acc_{name}'] = torch.mean(attack_word_accs.type(torch.float)).item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(orig_image_no_w_tensor,
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(orig_image_w_tensor,
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_w.png'), nrow=8)
            # save_image(utils_img.unnormalize_img(imgs),
            #            os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            # save_image(utils_img.unnormalize_img(imgs_w),
            #            os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_w.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
