# PnP-AE
The codes for the work "PnP-AE: A Plug-and-Play Module for Volumetric Medical Image Segmentation"
- have been accepted at **BIBM 2023**.
- Framework
    - ![](./figures/framework.png)

- Result on U-Net
    - ![](./figures/result.png)

## 1. Prepare data
- BTCV
      - The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).
- BraTS
     - [braTS2018](https://pan.baidu.com/s/1A7xRvpNZgl-tgDnpCvU1Pg)
          -  code：1234
     - [braTS2013](https://www.smir.ch/BRATS/Start2013)


## 2. Environment

- Use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Train
    ```
    For example (BTCV):
        - Unet++ add PnP-AE 
            python train.py --dataset Synapse --cfg configs/config_UnetPlusPlus.py --root_path data/SynapseII --max_epochs 150 --output_dir unet++25D_Synapse  --img_size 224 --base_lr 0.05 --batch_size 16 --enable_25D --min_test_epoch 150 --volume_path data/SynapseII/test_vol_h5
        - unet++ 2D
            python train.py --dataset Synapse --cfg configs/config_UnetPlusPlus.py --root_path    data/SynapseII --max_epochs 150 --output_dir unet++_Synapse  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 150 --volume_path data/SynapseII/test_vol_h5
    
    For example (BraTS):
        - Unet++ add PnP-AE 
            python train.py --dataset BRATS2018 --cfg configs/config_UnetPlusPlus.py --root_path data/BRATS2018 --max_epochs 150 --output_dir unet++25D_BRATS2018  --img_size 224 --base_lr 0.05 --batch_size 16 --enable_25D --min_test_epoch 150 --volume_path data/BRATS2018/test_vol_h5
        - unet++ 2D
            python train.py --dataset BRATS2018 --cfg configs/config_UnetPlusPlus.py --root_path data/BRATS2018 --max_epochs 150 --output_dir unet++_BRATS2018  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 150 --volume_path data/BRATS2018/test_vol_h5
    ```
    
- Test
    ```
    For example (BTCV):
        - Unet++ add PnP-AE 
            python test.py --dataset Synapse --cfg configs/config_UnetPlusPlus.py --is_saveni --volume_path data/Synapse --output_dir unet++25D --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 16 --enable_25D
        - unet++ 2D
            python test.py --dataset Synapse --cfg configs/config_UnetPlusPlus.py --is_saveni --volume_path data/Synapse --output_dir unet++ --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
    
    For example (BraTS):
        - Unet++ add PnP-AE 
            python test.py --dataset BRATS2018 --cfg configs/config_UnetPlusPlus.py --is_saveni --volume_path data/BRATS2018 --output_dir unet++25D_BRATS2018 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 16 --enable_25D
        - unet++ 2D
            python test.py --dataset BRATS2018 --cfg configs/config_UnetPlusPlus.py --is_saveni --volume_path data/BRATS2018 --output_dir unet++_BRATS2018 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
    ```



## 4. Supported models (Welcome to add new 2D neworks)
- √ U-Net
- √ U-Net++
- √ SwinUNet
- √ MissFormer
- √ TransUNet

## 5. Pretrain Model
- TransUNet pretrain (base offical)
    * [Get models in this link](https://console.cloud.google.com/storage/vit_models/):   R50-ViT-B_16, ViT-B_16, ViT-L_16...

- SwinUNet pretrain (base offical)
    * [Get pre-trained model in this link](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing)
