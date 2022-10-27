import numpy as np
import nibabel as nb
import glob
import cv2
import h5py
import os
from tqdm import tqdm
root_dir = "training/*/*"

files = glob.glob(root_dir)
labels = []
images = []
for each in files:
    if "frame" in each and "gt" in each:
        labels.append(each)
    elif "frame" in each:
        images.append(each)

os.makedirs('data/ACDC/train_npz',exist_ok=True)
os.makedirs('data/ACDC/train_npz_25D',exist_ok=True)
os.makedirs('data/ACDC/test_vol_h5',exist_ok=True)
os.makedirs('lists/lists_ACDC/',exist_ok=True)
last_patient = None
random_test_case = ['001','002','003','004','006','007','008','009','011','012',
                    '013','016','017','021','025','026','027','028','029','030']
train_file = open('lists/lists_ACDC/train.txt','w')
test_h5_file = open('lists/lists_ACDC/test_vol.txt','w')
for i in tqdm(range(len(images))):
    patient = images[i].split("\\")[-2]
    image = nb.load(images[i]).get_fdata()
    label = nb.load(labels[i]).get_fdata()
    image = image/255
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
    assert image.shape[2]==label.shape[2],f'{image.shape[2],label.shape[2],images[i],labels[i]}'
    slices = image.shape[2]
    if last_patient == patient:
        tag = '02'
    else:
        tag = '01'
        last_patient = patient
    if patient[-3:] not in random_test_case:
        for num in range(slices):
            #2D
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            np.savez("data/ACDC/train_npz/" + str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3),image = case_image, label=case_label)
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
            np.savez("data/ACDC/train_npz_25D/" + str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3),image = case_image25D, label=case_label)
            train_file.write(str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3)+'\n')
    else:
        image_h5, label_h5 = np.zeros((slices, 224, 224)), np.zeros((slices, 224, 224))
        for num in range(slices):
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_AREA)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            image_h5[num], label_h5[num] = case_image, case_label
        with h5py.File(f"data/ACDC/test_vol_h5/{patient}_{tag}.npy.h5", 'w') as f:
            f.create_dataset('image', data=image_h5)
            f.create_dataset('label', data=label_h5)
        test_h5_file.write(f'{patient}_{tag}'+'\n')
train_file.close()
test_h5_file.close()