import h5py
import os
import numpy as np
from glob import glob
files = glob('data\\Synapse\\test_vol_h5\\*')
new_path25D = 'data\\SynapseII\\train_npz_25D'
new_path2D = 'data\\SynapseII\\train_npz'
original_test = [f.split('/'[-1].split('.')[0].split('e')[-1]) for f in files]
wait_list = []
stride = 1
for i in range(6):
    wait_list.append(original_test[np.random.choice(len(original_test))])
f = open('lists/lists_SynapseII/train.txt', 'a+')
for file in files:
    if file[-11:-7] not in wait_list:
        continue
    # print(file)
    data = h5py.File(file)
    case = file.split('\\')[-1].split('.')[0]
    images, labels = data['image'][:], data['label'][:]
    for idx in range(len(images)):
        if idx<stride or idx>=len(images)-stride:
            image25D = [images[idx][...,np.newaxis], images[idx][...,np.newaxis], images[idx][...,np.newaxis]]
            label = labels[idx]
        else:
            image25D = [images[idx-1][...,np.newaxis], images[idx][...,np.newaxis], images[idx+1][...,np.newaxis]]
            label = labels[idx]
        image25D = np.concatenate(image25D, axis=-1)
        image2D = images[idx]
        case_slice = case+'_slice'+str(idx).zfill(3)+'.npz'
        new_file25D = os.path.join(new_path25D, case_slice)
        new_file2D = os.path.join(new_path2D, case_slice)
        # f.write(case+'_slice'+str(idx).zfill(3)+'\n')
        np.savez(new_file25D, image=image25D, label=label)
        np.savez(new_file2D, image=image2D, label=label)
f.close()
