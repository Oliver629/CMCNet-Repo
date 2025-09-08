# CMCNet

This repo holds code for [CMCNet:Cross-directional Morphology-aware Convolution Network for Chest X-ray Anatomy Segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0925231225017849)



## Usage

### 1. Prepare data (We use VinDr-RibCXR dataset)

[Please use 2D X-ray data for testing](https://vindr.ai/ribcxr)


### 2. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test

- Run the train script on dataset. 

```bash
CUDA_VISIBLE_DEVICES=0 python train.py 
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)



## Citations


```bibtex
@article{huang2025cmcnet,
  title={CMCNet: Cross-directional Morphology-aware Convolution Network for Chest X-ray Anatomy Segmentation},
  author={Huang, Lili and Feng, Yuhan and Zhao, Xiaowei and Li, Chenglong and Tang, Jin},
  journal={Neurocomputing},
  pages={131112},
  year={2025},
  publisher={Elsevier}
}
```
