import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import cv2

# parent_path = "unet25D_Synapse"
# parent_path = "unet_Synapse"
parent_path = "unet25D_BRATS2018"
# parent_path = "unet_BRATS2018"
case = "Brats18_2013_22_1"

classes = 3
# color_map = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,255,0],[255,128,0]]
color_map = [[255,0,0],[0,255,0],[0,0,255]]
# synapse = [[0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255], [255,255,0], [0,255,128], [0,128,255]]
# BraTs = [[0,0,255],[0,255,0],[255,0,0]]
os.makedirs(os.path.join(parent_path, 'results/img_gt'), exist_ok=True)
os.makedirs(os.path.join(parent_path, 'results/img_pred'), exist_ok=True)
# os.makedirs('unet_Synapse/results/img_25D', exist_ok=True)
os.makedirs(os.path.join(parent_path, 'results/img_2D'), exist_ok=True)
imgs = nib.load(os.path.join(parent_path, f'predictions/{case}_img.nii.gz')).get_fdata()
gts = nib.load(os.path.join(parent_path, f'predictions/{case}_gt.nii.gz')).get_fdata()
preds = nib.load(os.path.join(parent_path, f'predictions/{case}_pred.nii.gz')).get_fdata()

def transfrom(gt, color_map, n_class=8):
    for i in range(n_class):
        for j in range(3):
            gt[...,j][gt[...,j]==(i+1)] = color_map[i][j]
    return gt

for i in range(imgs.shape[2]):
    img = (imgs[:,:,i][:,:,np.newaxis].repeat(3,-1)*255).astype(np.uint8)
    # if i==0:
    #     img1 = imgs[:, :, i][:, :, np.newaxis]
    #     img2 = imgs[:, :, i][:, :, np.newaxis]
    #     img3 = imgs[:, :, i][:, :, np.newaxis]
    #     img25D = (np.concatenate([img1, img2, img3], axis=-1)*255).astype(np.uint8)
    # elif i==imgs.shape[2]-1:
    #     img1 = imgs[:, :, i][:, :, np.newaxis]
    #     img2 = imgs[:, :, i][:, :, np.newaxis]
    #     img3 = imgs[:, :, i][:, :, np.newaxis]
    #     img25D = (np.concatenate([img1, img2, img3], axis=-1)*255).astype(np.uint8)
    # else:
    #     img1 = imgs[:, :, i - 1][:, :, np.newaxis]
    #     img2 = imgs[:, :, i][:, :, np.newaxis]
    #     img3 = imgs[:, :, i + 1][:, :, np.newaxis]
    #     img25D = (np.concatenate([img1, img2, img3], axis=-1)*255).astype(np.uint8)

    gt = gts[:,:,i][:,:,np.newaxis].repeat(3,-1).astype(np.uint8)
    pred = preds[:,:,i][:,:,np.newaxis].repeat(3,-1).astype(np.uint8)
    gt_mask = (gt>0).astype(np.uint8)
    pred_mask = (pred>0).astype(np.uint8)
    # print(np.unique(gt))
    gt = transfrom(gt, color_map, n_class = classes)
    pred = transfrom(pred, color_map, n_class = classes)
    img_gt = (1-gt_mask)*img+gt*85
    img_pred = img*(1-pred_mask)+pred*85
    cv2.imwrite(os.path.join(parent_path, f'results/img_gt/{case}_{i}.jpg'), img_gt)
    cv2.imwrite(os.path.join(parent_path, f'results/img_pred/{case}_{i}.jpg'), img_pred)
    # cv2.imwrite(os.path.join(parent_path, f'results/img_25D/{case}_{i}.jpg'), img25D)
    cv2.imwrite(os.path.join(parent_path, f'results/img_2D/{case}_{i}.jpg'), img)

    # cv2.imshow('1', img_gt)
    # cv2.imshow('1', img_gt)
    # key = cv2.waitKey(0)
    # plt.subplot(1, 3, 1);
    # plt.imshow(img, cmap='bone');
    # plt.axis('OFF');
    # plt.title('image')

    # plt.subplot(1, 2, 1)
    # plt.imshow(img_gt)
    # plt.axis('OFF')
    # plt.title('gt')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(img, cmap='gray')
    # plt.imshow(pred)
    # plt.axis('OFF')
    # plt.title('pred')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'swin_unet_Synapse/results/patient009_01_{i}.jpg')
