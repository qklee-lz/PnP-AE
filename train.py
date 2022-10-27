import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet
from datasets.dataset_synapse import Synapse_dataset
from networks.Unet import UNet
from networks.Unet_plusplus import UNet_2Plus
from networks.MISSFormer import MISSFormer
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--min_test_epoch', type=int,
                    default=100, help='When to start doing testing')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--enable_25D', action='store_true', help='use 2.5D data and model')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--volume_path', type=str,
                    default='data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir

args = parser.parse_args()
if args.dataset == "Synapse":
    if args.enable_25D:
        args.root_path = os.path.join(args.root_path, "train_npz_25D")
    else:
        args.root_path = os.path.join(args.root_path, "train_npz")
elif args.dataset == "ACDC":
    if args.enable_25D:
        args.root_path = os.path.join(args.root_path, "train_npz_25D")
    else:
        args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'root_path': args.root_path,
            'list_dir': './lists/lists_SynapseII',
            'num_classes': 9,
        },
        'ACDC': {
            'Dataset': Synapse_dataset,
            'root_path': args.root_path,
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.enable_25D:
        tag = '2.5D'
    else:
        tag = '2D'
    if config.MODEL.NAME=='swin_tiny_patch4_window7_224':
        print(f'Using {tag} swin-unet')
        net = SwinUnet(args.enable_25D, config, img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(config)
    elif config.MODEL.NAME=='unet':
        print(f'Using {tag} unet')
        net = UNet(args.enable_25D, n_classes=args.num_classes, bilinear=config.MODEL.UNET.bilinear).cuda()
    elif config.MODEL.NAME=='unet_plusplus':
        print(f'Using {tag} unet_plusplus')
        net = UNet_2Plus(args.enable_25D, n_classes=args.num_classes, is_deconv=config.MODEL.UNET_PLUSPLUS.is_deconv,
                         is_batchnorm=config.MODEL.UNET_PLUSPLUS.is_batchnorm, is_ds=config.MODEL.UNET_PLUSPLUS.is_ds).cuda()
    elif config.MODEL.NAME=='MISSFormer':
        print(f'Using {tag} MISSFormer')
        net = MISSFormer(args.enable_25D, num_classes=args.num_classes).cuda()
    elif config.MODEL.NAME == 'TransUnet':
        print(f'Using {tag} TransUnet')
        config_vit = CONFIGS_ViT_seg[config.MODEL.TRANSUNET.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = config.MODEL.TRANSUNET.n_skip
        if config.MODEL.TRANSUNET.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / config.MODEL.TRANSUNET.vit_patches_size), int(args.img_size / config.MODEL.TRANSUNET.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        # print(config_vit.pretrained_path)
        net.load_from(weights=np.load(config_vit.pretrained_path))
        # print(config_vit.pretrained_path)
    trainer = {'Synapse': trainer_synapse,'ACDC': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)
#Synapse
#swin-unet 2.5D
# python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/SynapseII --max_epochs 150 --output_dir swin_unet25D  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
# python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/SynapseII --output_dir swin_unet25D --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#swin-unet 2D
#python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/SynapseII --max_epochs 150 --output_dir swin_unet  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
#python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/SynapseII --output_dir swin_unet --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#unet 2.5D
# python train.py --dataset Synapse --cfg configs/Unet.yaml --root_path data/SynapseII --max_epochs 150 --output_dir unet25D  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
# python test.py --dataset Synapse --cfg configs/Unet.yaml --is_saveni --volume_path data/SynapseII --output_dir unet25D --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#unet 2D
#python train.py --dataset Synapse --cfg configs/Unet.yaml --root_path data/SynapseII --max_epochs 150 --output_dir unet  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
#python test.py --dataset Synapse --cfg configs/Unet.yaml --is_saveni --volume_path data/SynapseII --output_dir unet --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#unet++ 2.5D
# python train.py --dataset Synapse --cfg configs/Unet_plusplus.yaml --root_path data/Synapse --max_epochs 150 --output_dir unet++25D  --img_size 224 --base_lr 0.05 --batch_size 16 --enable_25D
# python test.py --dataset Synapse --cfg configs/Unet_plusplus.yaml --is_saveni --volume_path data/Synapse --output_dir unet++25D --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 16 --enable_25D
#unet++ 2D
#python train.py --dataset Synapse --cfg configs/Unet_plusplus.yaml --root_path data/Synapse --max_epochs 150 --output_dir unet++  --img_size 224 --base_lr 0.05 --batch_size 24
#python test.py --dataset Synapse --cfg configs/Unet_plusplus.yaml --is_saveni --volume_path data/Synapse --output_dir unet++ --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#Missformer 2.5D
# python train.py --dataset Synapse --cfg configs/MissFormer.yaml --root_path data/SynapseII --max_epochs 150 --output_dir MissFormer25D  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
# python test.py --dataset Synapse --cfg configs/MissFormer.yaml --is_saveni --volume_path data/Synapse --output_dir MissFormer25D --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#Missformer 2D
#python train.py --dataset Synapse --cfg configs/MissFormer.yaml --root_path data/SynapseII --max_epochs 150 --output_dir MissFormer  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/SynapseII/test_vol_h5
#python test.py --dataset Synapse --cfg configs/MissFormer.yaml --is_saveni --volume_path data/Synapse --output_dir MissFormer --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#TransUnet 2.5D
# python train.py --dataset Synapse --cfg configs/TransUnet.yaml --root_path data/2D5/SynapseII --max_epochs 200 --output_dir TransUnet  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D
# python test.py --dataset Synapse --cfg configs/TransUnet.yaml --is_saveni --volume_path data/Synapse --output_dir TransUnet --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#TransUnet 2D
# python train.py --dataset Synapse --cfg configs/TransUnet.yaml --root_path data/2D5/SynapseII --max_epochs 200 --output_dir TransUnet  --img_size 224 --base_lr 0.05 --batch_size 24
# python test.py --dataset Synapse --cfg configs/TransUnet.yaml --is_saveni --volume_path data/Synapse --output_dir TransUnet --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24

#ACDC
#swin-unet 2.5D
# python train.py --dataset ACDC --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/ACDC --max_epochs 150 --output_dir swin_unet25D_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/ACDC --output_dir swin_unet25D_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#swin-unet 2D
#python train.py --dataset ACDC --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/ACDC --max_epochs 150 --output_dir swin_unet_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
#python test.py --dataset ACDC --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/ACDC --output_dir swin_unet_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#unet 2.5D
# python train.py --dataset ACDC --cfg configs/Unet.yaml --root_path data/ACDC --max_epochs 150 --output_dir unet25D_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/Unet.yaml --is_saveni --volume_path data/Synapse --output_dir unet25D_ACDC --max_epoch 300 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#unet 2D
#python train.py --dataset ACDC --cfg configs/Unet.yaml --root_path data/ACDC --max_epochs 150 --output_dir unet_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
#python test.py --dataset ACDC --cfg configs/Unet.yaml --is_saveni --volume_path data/Synapse --output_dir unet_ACDC --max_epoch 300 --base_lr 0.05 --img_size 224 --batch_size 24
#unet++ 2.5D
# python train.py --dataset ACDC --cfg configs/Unet_plusplus.yaml --root_path data/ACDC --max_epochs 150 --output_dir unet++25D_ACDC  --img_size 224 --base_lr 0.05 --batch_size 12 --enable_25D --min_test_epoch 130 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/Unet_plusplus.yaml --is_saveni --volume_path data/Synapse --output_dir unet++25D_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 16 --enable_25D
#unet++ 2D
#python train.py --dataset ACDC --cfg configs/Unet_plusplus.yaml --root_path data/ACDC --max_epochs 150 --output_dir unet++_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 130 --volume_path data/ACDC/test_vol_h5
#python test.py --dataset ACDC --cfg configs/Unet_plusplus.yaml --is_saveni --volume_path data/Synapse --output_dir unet++_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#Missformer 2.5D
# python train.py --dataset ACDC --cfg configs/MissFormer.yaml --root_path data/ACDC --max_epochs 150 --output_dir MissFormer25D_ACDC  --img_size 224 --base_lr 0.05 --batch_size 12 --enable_25D --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/MissFormer.yaml --is_saveni --volume_path data/ACDC --output_dir MissFormer25D_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#Missformer 2D
#python train.py --dataset ACDC --cfg configs/MissFormer.yaml --root_path data/ACDC --max_epochs 150 --output_dir MissFormer_ACDC  --img_size 224 --base_lr 0.05 --batch_size 12 --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
#python test.py --dataset ACDC --cfg configs/MissFormer.yaml --is_saveni --volume_path data/ACDC --output_dir MissFormer_ACDC --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
#TransUnet 2.5D
# python train.py --dataset ACDC --cfg configs/TransUnet.yaml --root_path data/ACDC --max_epochs 150 --output_dir TransUnet25D_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --enable_25D --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/TransUnet.yaml --is_saveni --volume_path data/ACDC --output_dir TransUnet --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24 --enable_25D
#TransUnet 2D
# python train.py --dataset ACDC --cfg configs/TransUnet.yaml --root_path data/ACDC --max_epochs 150 --output_dir TransUnet_ACDC  --img_size 224 --base_lr 0.05 --batch_size 24 --min_test_epoch 100 --volume_path data/ACDC/test_vol_h5
# python test.py --dataset ACDC --cfg configs/TransUnet.yaml --is_saveni --volume_path data/ACDC --output_dir TransUnet --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24


