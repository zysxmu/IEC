
# Test-Time Iterative Error Correction for Efficient Diffusion Models (ICLR2026)

## Introduction
This repository contains the official PyTorch implementation for the ICLR2026 paper
*["Test-Time Iterative Error Correction for Efficient Diffusion Models"](https://arxiv.org/abs/2511.06250).

## Before Start

Our code is high based on CacheQuant (CVPR2025). We high appreacite their contribution!

```
@article{liu2025cachequant,
      title={CacheQuant: Comprehensively Accelerated Diffusion Models}, 
      author={Xuewen Liu and Zhikai Li and Qingyi Gu},
      journal={arXiv},
      year={2025}
}
```

## Preparation

### Environment
Create and activate a suitable conda environment named `E` by using the following command:

```bash
cd IEC
conda env create -f environment.yaml
conda activate IEC
```

### Pretrained Model and Data

Pre-trained models for DDPM are automatically downloaded. 

For LDM experiments, download pre-trained models to `mainldm/models/ldm` following the instructions in the *[latent-diffusion](https://github.com/CompVis/latent-diffusion#model-zoo)* and *[stable-diffusion](https://github.com/CompVis/stable-diffusion#weights)* repos. 


Please download original datasets used for evaluation from each dataset’s official website.


## Usage

Here, we proivde an example to apply our IEC to the CacheQuant. Note the IEC has a very simple code as shown in the function 'p_sample_ddim_implicit_2' in './mainldm/ldm/models/diffusion/ddim.py'.

1. Obtain DPS and Calibration
```bash
python ./mainldm/sample_cachequant_imagenet_cali.py
```
2. Get Cached Features
```bash
python ./mainldm/sample_cachequant_imagenet_predadd.py
```
3. Calculate DEC for Cache
```bash
python ./err_add/imagenet/cache_draw.py --error cache
```
4. Get Quantized Parameters
```bash
python ./mainldm/sample_cachequant_imagenet_params.py
```
5. Calculate DEC for Quantization
```bash
python ./err_add/imagenet/cache_draw.py --error quant
```
6. Acceleration and Sample
```bash
python ./mainldm/sample_cachequant_imagenet_quant.py <--recon>
```
The `--recon` to use reconstruction.

### Details
The repo provides code for all experiments. We use the LDM-4 on ImageNet as an example to illustrate the usage. Other experiments are implemented similarly. Our experiments are aligned with *[Deepcache](https://github.com/horseee/DeepCache)*: non-uniform caching is used for stable-diffusion, and for other models only when `intervel` is greater than 10. We use the *[guided-diffusion](https://github.com/openai/guided-diffusion)* and *[clip-score](https://github.com/Taited/clip-score)* to evaluate results. The accelerated diffusion models are deployed by utilizing *[CUTLASS](https://github.com/NVIDIA/cutlass)* and *[torch_quantizer](https://github.com/ThisisBillhe/torch_quantizer)*.


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhong2025test,
  title={Test-Time Iterative Error Correction for Efficient Diffusion Models},
  author={Zhong, Yunshan and Qi, Yanwei and Zhang, Yuxin},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```


