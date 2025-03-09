# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import json
import os
import time
import random

from PIL import Image
import numpy as np
from pathlib import Path
import warnings

import torch_dct
from torchvision.transforms import functional
from torchvision.utils import save_image
import encoder_unet
import data_augmentation


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
    mask = np.zeros(size, dtype=np.float32)
    # 生成同心圆形状的掩码
    for y in range(size[0]):
        for x in range(size[1]):
            # calculate the distance between the center of circle and the target points
            distance_to_center = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
            # the area in the concentric circles
            if radius <= distance_to_center < radius_2:
                mask[y, x] = num_masked
            # the area not in the concentric circles
            else:
                mask[y, x] = num_unmasked
    return mask


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="../stable_signature/dataset/test2014")
    # aa("--train_dir", type=str, default="Val_att")

    aa("--val_dir", type=str, default="../stable_signature/dataset/coco-mini-mini-test-copy")

    group = parser.add_argument_group('Marking parameters')
    aa("--num_bits", type=int, default=4, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=256, help="Image size")
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
    aa("--epochs", type=int, default=1, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--lambda_w", type=float, default=0.8, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss. (Default: 0.0)")
    aa("--loss_margin", type=float, default=1,
       help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='mse',
       help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce',
       help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=1, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=2, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=8, help="Number of workers for data loading. (Default: 8)")

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

    # aa("--p_crop", type=float, default=-0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    # aa("--p_res", type=float, default=-0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    # aa("--p_blur", type=float, default=-0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    # aa("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    # aa("--p_rot", type=float, default=-0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    # aa("--p_color_jitter", type=float, default=-0.5,
    #    help="Probability of the color jitter augmentation. (Default: 0.5)")

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)
    aa('--dist', type=utils.bool_inst, default=False, help='Enabling distributed training')

    group = parser.add_argument_group('Misc')
    aa('--seed', default=2, type=int, help='Random seed')
    aa('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    # aa('--dataset', default='coco')
    aa('--num_images', default=1, type=int)
    # aa('--guidance_scale', default=1, type=float)
    aa('--guidance_scale', default=7.5, type=float)
    aa('--num_inference_steps', default=50, type=int)
    aa('--image_length', default=512, type=int)

    return parser


def main(params):
    import ssl
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            n_channels=2, n_classes=2, num_bits=params.num_bits, bilinear=True)
    else:
        raise ValueError('Unknown encoder type')
    encoder = encoder.to(device)
    print('\nencoder: \n%s' % encoder)
    print('total parameters: %d' % sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits * params.redundancy,
                                       channels=params.decoder_channels, input_channel=2)
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
                                               params.p_rot, params.p_color_jitter, params.p_res).to(device)
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
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    print('training...')
    start_time = time.time()
    best_bit_acc = 0

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_id='runwayml/stable-diffusion-v1-5'
    # model_id = 'stabilityai/stable-diffusion-2-1-base'
    model_id='stabilityai/stable-diffusion-2-base'

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')

    pipeline = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    )

    pipeline = pipeline.to(device)

    for epoch in range(start_epoch, params.epochs):
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer, scheduler, epoch, params, pipe=pipeline,
                                      data_aug=data_aug)
    return 0


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


def train_one_epoch(encoder_decoder: models.EncoderDecoder, loader, optimizer, scheduler, epoch, params, pipe, data_aug,
                    is_mask=True):
    """
    One epoch of training.
    """
    warnings.filterwarnings("ignore")

    if params.scheduler is not None:
        scheduler.step(epoch)

    encoder_decoder.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")

    total_step = params.total_step
    watermarked_step = params.start_step

    # get text embedding
    dataset, prompt_key = get_dataset(params)
    tester_prompt = ''  # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    torch.autograd.set_detect_anomaly(True)
    stop_step = params.start_step
    for it, (_, _) in enumerate(metric_logger.log_every(loader, 10, header)):
        init_latents_no_w = pipe.get_random_latents(height=params.image_length, width=params.image_length)

        print('it:', it)

        # current_prompt = dataset[it][prompt_key]

        current_prompt = 'realistic painting of a tardigrade kaiju, with 6 legs in a desert storm, by james gurney, slime, big globule eye, godzilla, vintage, concept art, oil painting, tonalism, crispy'

        # change the param stop_step to determine the injecting watermark timestep
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

        # save tensor from latent code without wtermark
        tensor_path = f'stage1_35/ori_latent/{it}.pt'
        # tensor_path = f'stage1_35/ori_latent/{it}.pt'
        # tensor_path = f'stable_v1.5/ori_latent/{it+286}.pt'
        torch.save(outputs_latents_no_w, tensor_path)
    return 0


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
