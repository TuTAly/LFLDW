import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
# from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_step = 50


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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


def fft_image(image):
    # 对图像的最后两个维度进行二维傅里叶变换
    image_fft = torch.fft.fft2(image)
    image_fft = torch.fft.fftshift(image_fft)
    # image_fft = torch.abs(image_fft)
    # 返回复数形式的傅里叶变换结果
    return image_fft


def ifft_image(image_fft):
    # 对复数形式的傅里叶变换结果进行逆变换
    image_fft = torch.fft.ifftshift(image_fft)
    image = torch.fft.ifft2(image_fft)
    # 返回逆变换后的实部
    return image


@torch.no_grad()
def invert(start_latents, batch, pipe, prompt=None, negative_prompt=None, guidance_scale=1, num_inference_steps=50,
           num_images_per_prompt=1, do_classifier_free_guidance=True, is_final=False, device=device):
    # Encode prompt

    prompt = []
    negative_prompt = []

    for i in range(batch):
        prompt.append('')
        negative_prompt.append('')

    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )

    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    if is_final == False:
        aa = num_inference_steps - 1
    else:
        aa = num_inference_steps

    for i in range(1, num_inference_steps):

        # We'll skip the final iteration
        # if i >= num_inference_steps - 1: continue
        if i >= aa: continue

        t = timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        # the problem is here
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # noise_pred = pipe.unet(latent_model_input, t,encoder_hidden_states=text_embeddings,batch=2).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)
    # print_trainable_parameters(pipe.unet)

    return torch.stack(intermediate_latents)


# Sample function (regular DDIM)
@torch.no_grad()
def sample_from_latents(batch, pipe, start_latents=None,
                        guidance_scale=1, num_inference_steps=30,
                        num_images_per_prompt=1, do_classifier_free_guidance=True,
                        device=device):
    prompt = []
    negative_prompt = []

    for i in range(batch):
        prompt.append('')
        negative_prompt.append('')

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    # for i in range(start_step, num_inference_steps):
    #     # for i in tqdm(range(start_step, num_inference_steps)):
    #
    #     t = pipe.scheduler.timesteps[i]
    #
    #     # expand the latents if we are doing classifier free guidance
    #     latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #     latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    #
    #     # predict the noise residual
    #     # !!!!!!!!                Here is the problem           !!!!!!!!!!!!!!!!!!
    #
    #     noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    #
    #     # perform guidance
    #     if do_classifier_free_guidance:
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #
    #     # Normally we'd rely on the scheduler to handle the update step:
    #     # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    #
    #     # Instead, let's do it ourselves:
    #     prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
    #     alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
    #     alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
    #     predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    #     direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
    #     latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    #
    # # Post-processing
    # print('!!!!!!!!!!!!!!!!!!!!!!number!!!!!!!!!!!!!')
    # print_trainable_parameters(pipe.unet)

    latents = latents / 0.18215
    images = pipe.vae.decode(latents)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # change channel in the images
    images = images.permute(0, 2, 3, 1)

    # 将RGB通道的顺序改为GBR
    # images = images[..., [0, 2, 1]]

    # 如果需要，将通道维度调整回原始位置
    images = images.permute(0, 3, 1, 2)

    # images = pipe.decode_latents(latents)
    # images = pipe.numpy_to_pil(images)

    return images


# Sample function (regular DDIM)
@torch.no_grad()
def sample(batch, pipe, start_step=0, start_latents=None,
           guidance_scale=1, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           device=device):
    prompt = []
    negative_prompt = []

    for i in range(batch):
        prompt.append('')
        negative_prompt.append('')

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in range(start_step, num_inference_steps):
        # for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        # !!!!!!!!                Here is the problem           !!!!!!!!!!!!!!!!!!

        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    # print('!!!!!!!!!!!!!!!!!!!!!!number!!!!!!!!!!!!!')
    # print_trainable_parameters(pipe.unet)

    latents = latents / 0.18215
    images = pipe.vae.decode(latents)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16

    return images


def load_image(url, size=None):
    # response = requests.get(url,timeout=0.2)
    # img = Image.open(BytesIO(response.content)).convert('RGB')
    img = Image.open(url).convert('RGB')

    if size is not None:
        img = img.resize(size)
    return img


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_bits = 48
    imgs = torch.rand(size=[4, 3, 224, 224]).to(device)
    msgs = torch.rand(size=[16, num_bits]).to(device)

    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    im = load_image('imgs/00.png')
    # im = im.resize((256, 256))  # resize for convenient viewing
    plt.title('original image')
    plt.imshow(im)
    plt.show()
    im.save('temp/original.png')

    with torch.no_grad():
        im = tfms.functional.to_tensor(im)
        im = im.unsqueeze(0)
        im = im.to(device)
        im = torch.cat((im, im), dim=0)
        latent = pipe.vae.encode(im * 2 - 1)

    # !!!!!! here is the problem !!!!!!!!!!!!!!!!!!!!!
    l = 0.18215 * latent.latent_dist.sample()
    a = transforms.ToPILImage()(l[1])
    a.save('temp/latent.png')

    a = sample_from_latents(pipe=pipe, start_latents=l, batch=2, num_inference_steps=total_step)[0]
    a = transforms.ToPILImage()(a)
    plt.title('original inverted image without diffusion')
    plt.imshow(a)
    plt.show()

    # input_image_prompt = "Beautiful DSLR Photograph of a penguin on the beach, golden hour"
    input_image_prompt = ['', '']
    # inverted_latents = invert(l[0][None], input_image_prompt, num_inference_steps=total_step)
    inverted_latents = invert(l, batch=2, pipe=pipe,
                              num_inference_steps=total_step)  # output: Tensor [inversion time step, batch, c, w, h]

    # start_step = int(total_step / 5 * 1 - 2)
    start_step = int(total_step / 5 * 0)


    target_latents = inverted_latents[-(start_step + 1)]  # output: Tensor [batch, c, w, h]

    # target_latents = torch.cat((target_latents, target_latents), dim=0)

    a = sample(pipe=pipe, start_latents=target_latents, batch=2, start_step=start_step, num_inference_steps=total_step)[
        0]

    # temp = transforms.ToPILImage()(target_latents[0])  # 自动转换为0-255
    # temp.save('F/latent_space.png')
    a = transforms.ToPILImage()(a)
    plt.title('original inverted image')
    plt.imshow(a)
    plt.show()

    target_latents = inverted_latents[-(start_step + 1)]  # output: Tensor [batch, c, w, h]
    target_latents = target_latents.cpu().clone()
    dft_shift = fft_image(target_latents)

    # temp = transforms.ToPILImage()(torch.real(dft_shift[0]))  # 自动转换为0-255
    # temp.save('F/latent_space_F.png')
    # plt.imshow(temp)
    # plt.show()

    # ADD middle Freauency pass MASK

    batch, c, rows, cols = dft_shift.shape
    print('$$$$$$$$$$$$$$$$$$$shape:', batch, c, rows, cols)
    boundary_scale = 0.1
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    # mask = torch.zeros(c, rows, cols)
    circle = create_concentric_circle_mask(size=(rows, cols), center=(crow, ccol), radius=0, radius_2=20, num_masked=1,
                                           num_unmasked=0.5)
    # mask[:, :, :] = 0.5
    # mask[0, crow - 5:crow + 5, ccol - 5:ccol + 5] = 1
    # mask[1:, :, :] = 1

    dft_shift = dft_shift.numpy()
    dft_shift_mask = circle * dft_shift
    dft_shift_mask = torch.from_numpy(dft_shift_mask).to(device)

    # dft_shift_mask_real = torch.from_numpy(circle) * dft_shift.real
    # dft_shift_mask = dft_shift
    # dft_shift_mask.real = dft_shift_mask_real
    # dft_shift_mask = dft_shift_mask.to(device)

    iimg = ifft_image(dft_shift_mask)
    iimg = torch.real(iimg).to(device)
    # test whether the two tensor is same after Fourier Transform
    # target_latents = target_latents.to(device)
    # print(target_latents==iimg)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # target_latents[:, 0:, :, :] = iimg

    a = sample(pipe=pipe, start_latents=iimg, batch=2, start_step=start_step, num_inference_steps=total_step)[0]

    temp = transforms.ToPILImage()(a)  # 自动转换为0-255
    temp.save('F/latent_space_inverted.png')
    plt.title('inverted image after middle frequency pass filter')
    plt.imshow(temp)
    plt.show()

    print('finished')
