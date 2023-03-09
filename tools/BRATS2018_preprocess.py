import numpy as np
import nibabel as nb
import glob
import cv2
import h5py
import os
from tqdm import tqdm
import random
random.seed(2022)
root_dir = "data/MICCAI_BraTS_2018_Data_Training/*/*/*"

os.makedirs('data/BRATS2018/train_npz',exist_ok=True)
os.makedirs('data/BRATS2018/train_npz_25D',exist_ok=True)
os.makedirs('data/BRATS2018/test_vol_h5',exist_ok=True)
os.makedirs('lists/lists_BRATS2018/',exist_ok=True)
train_file = open('lists/lists_BRATS2018/train.txt', 'w')
test_h5_file = open('lists/lists_BRATS2018/test_vol.txt', 'w')

files = glob.glob(root_dir)
labels = []
images1 = []
images2 = []
images3 = []
for each in files:
    if "flair" in each and "nii" in each and '2013' in each:
        images1.append(each)
    elif "t2" in each and 'nii' in each and '2013' in each:
        images2.append(each)
    elif "t1ce" in each and 'nii' in each and '2013' in each:
        images3.append(each)
    elif "seg" in each and 'nii' in each and '2013' in each:
        labels.append(each)
test_case = []
for i in range(6):
    test_case.append((images1[np.random.choice(len(images1))]).split['/'][-1].split['.'][0])
images1.sort(key=lambda x :x.split(os.path.sep)[-2])
images2.sort(key=lambda x :x.split(os.path.sep)[-2])
images3.sort(key=lambda x :x.split(os.path.sep)[-2])
labels.sort(key=lambda x :x.split(os.path.sep)[-2])

for i in tqdm(range(len(images1))):
    case = images1[i].split(os.path.sep)[-2]
    image1 = nb.load(images1[i]).get_fdata().astype(np.float32)
    # image2 = nb.load(images2[i]).get_fdata().astype(np.float32)
    # image3 = nb.load(images3[i]).get_fdata().astype(np.float32)
    image = (image1 - image1.min()) / (image1.max() - image1.min())
    # image2 = (image2 - image2.min()) / (image2.max() - image2.min())
    # image3 = (image3 - image3.min()) / (image3.max() - image3.min())
    # image = image1+image2+image3
    # image = (image - image.min()) / (image.max() - image.min())
    label = nb.load(labels[i]).get_fdata()
    label[label==4] = 3
    label[label==1] = 4
    label[label==2] = 1
    label[label==4] = 2
    print(image.max(), np.unique(np.array(label)))
    size = image.shape
    if size[0]>size[1]:
        pad_left = (size[0]-size[1])//2+(size[0]-size[1])%2
        pad_right = (size[0]-size[1])//2
        image = np.pad(image,((0,0),(pad_left,pad_right),(0,0)),'constant')
        label = np.pad(label,((0,0),(pad_left,pad_right),(0,0)),'constant')
    elif size[0]<size[1]:
        pad_up = (size[1]-size[0])//2+(size[1]-size[0])%2
        pad_down = (size[1]-size[0])//2
        image = np.pad(image,((pad_up,pad_down),(0,0),(0,0)),'constant')
        label = np.pad(label,((pad_up,pad_down),(0,0),(0,0)),'constant')

    assert image.shape[0]==image.shape[1],'padding failed'
    assert image.shape[2]==label.shape[2],f'{image.shape[2],label.shape[2],images1[i],labels[i]}'
    #remove black image
    img = image.copy()
    lab = label.copy()
    image = []
    label = []
    for num in range(img.shape[2]):
        if img[...,num].sum() < 50:
            continue
        else:
            image.append(img[...,num][...,np.newaxis])
            label.append(lab[...,num][...,np.newaxis])
    image = np.concatenate(image, axis=-1)
    label = np.concatenate(label, axis=-1)
    slices = image.shape[2]
    print(image.shape)
    if case not in test_case:
        for num in range(slices):
            #2D
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            # c = case_image*0.5+(case_label/4)*0.5
            # cv2.imshow('1', c)
            # cv2.imshow('2', case_image)
            # key = cv2.waitKey(0)
            np.savez("data/BRATS2018/train_npz/" + case + "_slice" + str(num).zfill(3),image = case_image, label=case_label)
            #2.5D
            if num<1:
                case_image1 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image2 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image3 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image25D = np.concatenate([case_image1,case_image2,case_image3], axis=-1)
            elif num>slices-2:
                case_image1 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image2 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image3 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image25D = np.concatenate([case_image1,case_image2,case_image3], axis=-1)
            else:
                case_image1 = cv2.resize(image[:,:,num-1],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image2 = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image3 = cv2.resize(image[:,:,num+1],(224,224),interpolation=cv2.INTER_AREA)[...,np.newaxis]
                case_image25D = np.concatenate([case_image1,case_image2,case_image3], axis=-1)
            np.savez("data/BRATS2018/train_npz_25D/" + case + "_slice" + str(num).zfill(3),image = case_image25D, label=case_label)
            train_file.write(case + "_slice" + str(num).zfill(3)+'\n')
    else:
        image_h5, label_h5 = np.zeros((slices, 224, 224)), np.zeros((slices, 224, 224))
        for num in range(slices):
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            image_h5[num], label_h5[num] = case_image, case_label
        with h5py.File(f"data/BRATS2018/test_vol_h5/{case}.npy.h5", 'w') as f:
            f.create_dataset('image', data=image_h5)
            f.create_dataset('label', data=label_h5)
        test_h5_file.write(f'{case}'+'\n')
train_file.close()
test_h5_file.close()