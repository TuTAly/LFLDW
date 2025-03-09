from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler

from modified_stable_diffusion import ModifiedStableDiffusionPipeline


### credit to: https://github.com/cccntu/efficient-prompt-to-prompt

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
            alpha_tm1 ** 0.5
            * (
                    (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
                    + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler,
                 safety_checker,
                 feature_extractor,
                 requires_safety_checker: bool = True,
                 ):
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                                                                text_encoder,
                                                                tokenizer,
                                                                unet,
                                                                scheduler,
                                                                safety_checker,
                                                                feature_extractor,
                                                                requires_safety_checker)

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.forward_diffusion_2 = partial(self.backward_diffusion_2, reverse_process=True)

    def get_random_latents(self, latents=None, batch=1, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = batch
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    # @torch.inference_mode()
    # def backward_diffusion(
    #     self,
    #     use_old_emb_i=25,
    #     text_embeddings=None,
    #     old_text_embeddings=None,
    #     new_text_embeddings=None,
    #     latents: Optional[torch.FloatTensor] = None,
    #     num_inference_steps: int = 50,
    #     guidance_scale: float = 7.5,
    #     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #     callback_steps: Optional[int] = 1,
    #     reverse_process: True = False,
    #     **kwargs,
    # ):
    #     """ Generate image from text prompt and latents
    #     """
    #     # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    #     # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    #     # corresponds to doing no classifier free guidance.
    #     do_classifier_free_guidance = guidance_scale > 1.0
    #     # set timesteps
    #     self.scheduler.set_timesteps(num_inference_steps)
    #     # Some schedulers like PNDM have timesteps as arrays
    #     # It's more optimized to move all timesteps to correct device beforehand
    #     timesteps_tensor = self.scheduler.timesteps.to(self.device)
    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * self.scheduler.init_noise_sigma
    #
    #     if old_text_embeddings is not None and new_text_embeddings is not None:
    #         prompt_to_prompt = True
    #     else:
    #         prompt_to_prompt = False
    #
    #
    #     for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
    #         if prompt_to_prompt:
    #             if i < use_old_emb_i:
    #                 text_embeddings = old_text_embeddings
    #             else:
    #                 text_embeddings = new_text_embeddings
    #
    #         # expand the latents if we are doing classifier free guidance
    #         latent_model_input = (
    #             torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #         )
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #
    #         # predict the noise residual
    #         noise_pred = self.unet(
    #             latent_model_input, t, encoder_hidden_states=text_embeddings
    #         ).sample
    #
    #         # perform guidance
    #         if do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (
    #                 noise_pred_text - noise_pred_uncond
    #             )
    #
    #         prev_timestep = (
    #             t
    #             - self.scheduler.config.num_train_timesteps
    #             // self.scheduler.num_inference_steps
    #         )
    #         # call the callback, if provided
    #         if callback is not None and i % callback_steps == 0:
    #             callback(i, t, latents)
    #
    #         # ddim
    #         alpha_prod_t = self.scheduler.alphas_cumprod[t]
    #         alpha_prod_t_prev = (
    #             self.scheduler.alphas_cumprod[prev_timestep]
    #             if prev_timestep >= 0
    #             else self.scheduler.final_alpha_cumprod
    #         )
    #         if reverse_process:
    #             alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
    #         latents = backward_ddim(
    #             x_t=latents,
    #             alpha_t=alpha_prod_t,
    #             alpha_tm1=alpha_prod_t_prev,
    #             eps_xt=noise_pred,
    #         )
    #     return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i: i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def backward_diffusion(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            if_list=False,
            p=0.93,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        latents_list = []

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

            latents_list.append(latents)

        if if_list == True:
            return torch.stack(latents_list)
        else:
            return latents

    @torch.inference_mode()
    def backward_diffusion_2(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            x: Optional[torch.FloatTensor] = None,
            y: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            if_list=False,
            p=0.93,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler

        x = x * self.scheduler.init_noise_sigma
        y = y * self.scheduler.init_noise_sigma

        latents_list = []

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            x_model_input = (
                torch.cat([x] * 2) if do_classifier_free_guidance else x
            )

            y_model_input = (
                torch.cat([y] * 2) if do_classifier_free_guidance else y
            )

            x_model_input = self.scheduler.scale_model_input(x_model_input, t)
            y_model_input = self.scheduler.scale_model_input(y_model_input, t)

            y_inter_tp1 = (y_model_input - (1 - p) * x_model_input) / p
            x_inter_tp1 = (x_model_input - (1 - p) * y_inter_tp1) / p

            # predict the noise residual
            y_noise_pred = self.unet(
                x_inter_tp1, t, encoder_hidden_states=text_embeddings
            ).sample  # eposilon(x_inter_tp1 , t+1)

            x_noise_pred = self.unet(
                y_noise_pred, t, encoder_hidden_states=text_embeddings
            ).sample  # eposilon(y_tp1 , t+1)

            # perform guidance
            if do_classifier_free_guidance:
                x_noise_pred_uncond, x_noise_pred_text = x_noise_pred.chunk(2)
                x_noise_pred = x_noise_pred_uncond + guidance_scale * (
                        x_noise_pred_text - x_noise_pred_uncond
                )

                y_noise_pred_uncond, y_noise_pred_text = y_noise_pred.chunk(2)
                y_noise_pred = y_noise_pred_uncond + guidance_scale * (
                        y_noise_pred_text - y_noise_pred_uncond
                )

            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, x)
                callback(i, t, y)

            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            # 使用edict的方法

            # y_0=x_0

            y_tp1 = backward_ddim(
                x_t=y_inter_tp1,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=y_noise_pred,
            )

            x_tp1 = backward_ddim(
                x_t=x_inter_tp1,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=y_tp1,
            )

            latents = x_tp1

            latents_list.append(latents)

        if if_list == True:
            return torch.stack(latents_list)
        else:
            return latents

    @torch.inference_mode()
    def backward_diffusion_2(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            x: Optional[torch.FloatTensor] = None,
            y: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

