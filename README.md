# Localised Frequency Latent Domain Watermarking of DDIM Generated Images

This code is the official implementation of [LFLDW](https://ieeexplore.ieee.org/document/10890047).

If you have any questions, please contact us. (<xwj504@york.ac.uk>).

## Abstract
Stable Diffusion models, relying on iterative generative latent diffusion processes, have recently achieved remarkable results in producing realistic and diverse images. Meanwhile, the widespread application of generative models raised significant concerns about the origins of image content or the infringement of intellectual property rights. Consequently, a method for identifying AI generated images and/or other information about their origins is imperatively necessary. To address these requirements we propose to embed watermarks during one of the diffusion iterative steps of the DDIM. Such watermarks are required to be recoverable while also robust to possible changes to the generated watermarked images. The watermarks are embedded in the localized regions of the latent space frequencies. The binary watermarks are detected from the generated watermarked images by means of a CNN watermark detector. The robustness of the CNN watermark detector is improved through training by considering various distortions to the watermarked images.

## Usage

### Prepare Conda Environment
We have a `environment.yml` file, you can simply run:
```
conda env create -f environment.yml
```
### Prepare Dataset
Although our training process does not require any image training dataset, the code need to generate some data following the number of images in dataset. Please prepare any image dataset derectory named `train` and `test` (we can also name them freely).

### Download Checkpoint
Please go to [HuggingFace](https://huggingface.co/QR504/LFLDW/tree/main) and download the checkpoint of the model `checkpoint.pth`. Then, put in in the directory `V06.27`(we can also name them freely).

Note: You can also download some intermediate latent codes in the following training process. However, we recommend you to generate it by yourself by following `Training` process. The number of latent codes is only 265, which is limited in the training. We recommend that you only use these to debug code and familiarize yourself with the process.

### Training
We have a end-to-end training process, but it need more time because of diffusion inference process. We only recommend you to use it only in the finetune process.
```
python main_func_fft_end2end.py --train_dir train --val_dir test --output_dir V06.27/
```
#### Recommended Training
Generate latent codes in the intermediate denoising processes.
```
python main_func_fft_stage1.py --train_dir train
```
Use latent codes to train and it can save the time of generating intermediate latent codes.
```
python main_func_fft_stage2.py --train_dir train --val_dir test --output_dir V06.27/
```
### Test
In the inference process, you can run:
```
python watermark_test.py --train_dir train --val_dir test --output_dir V06.27/
```

## Parameters
Crucial hyperparameters which you can change for LFLDW:

- `r1` `r2`: the radius of the watermark region.
- `start_step`: watermark timestep in the diffusion denoising process.
- `p_crop`, `p_res`, `p_blur` etc. : attacks used in the distortion layer in the training.
- `w_weight`: watermark strength in the latent codes.

## Acknowledgements
Our code is based on the following repositories:
https://github.com/facebookresearch/stable_signature

https://github.com/YuxinWenRick/tree-ring-watermark

Thanks for their sharing.

## Citation
```
@INPROCEEDINGS{10890047,
  author={Lai, Qiran and Bors, Adrian G.},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Localised Frequency Latent Domain Watermarking of DDIM Generated Images}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Frequency-domain analysis;Diffusion processes;Watermarking;Detectors;Intellectual property;Robustness;Iterative methods;Speech processing;Protection;Image Generation;Digital Watermarking;Denoising Diffusion Implicit Model;Copyright protection},
  doi={10.1109/ICASSP49660.2025.10890047}}
```

## Suggestions are welcome! If the code is hard to read, we are sorry about it. Please contact us and ask us in issue. Please help us to make codes and tutorials better to read.

