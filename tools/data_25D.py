import numpy as np
from glob import glob
import os
from tqdm import tqdm
stride = 3
os.makedirs('data/SynapseII/train_npz_25D_s3', exist_ok=True)
def get_sort(x):
    case = int(x.split('\\')[-1].split('_')[0][-4:])
    slice = int(x.split('\\')[-1].split('.')[0][-3:])
    return case*1000+slice
train_data = glob('data\\SynapseII\\train_npz\\*')
train_data.sort(key=get_sort)
for file in tqdm(train_data):
    slice_idx = file.split('slice')[-1].split('.')[0]
    if not (os.path.exists(file.replace('slice'+slice_idx, 'slice'+str(int(slice_idx)+stride).zfill(3))) and os.path.exists(file.replace('slice'+slice_idx, 'slice'+str(int(slice_idx)-stride).zfill(3)))):
        files = [file for _ in range(3)]
    else:
        files = [file.replace('slice'+slice_idx, 'slice'+str(int(slice_idx)+_).zfill(3)) for _ in range(-stride, stride+1, stride)]
    new_image = []
    for f in files:
        data = np.load(f)
        image = data['image']
        image = image[...,np.newaxis]
        new_image.append(image)
    new_image = np.concatenate(new_image, axis=-1)
    data = np.load(file)
    new_label = data['label']
    new_file  = file.replace('train_npz', 'train_npz_25D_s3')
    # print(files)
    np.savez(new_file, image=new_image, label=new_label)

