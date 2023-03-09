import glob
import os
import h5py
train_path = 'lists/lists_BRATS2018/train.txt'
test_path = 'lists/lists_BRATS2018/test_vol.txt'
test_data_path = 'data/BRATS2018/test_vol_h5'
file_train = open(train_path, 'r')
file_test = open(test_path, 'r')
train_list = []
test_list = []

start_idx = 0
last_idx = 0
for file in file_train.readlines():
    current_idx = int(file.strip().split('slice')[-1])
    if current_idx<last_idx:
        train_list.append(last_idx-start_idx+1)
        start_idx = current_idx
    last_idx = current_idx

for file in file_test.readlines():
    filepath = test_data_path + "/{}.npy.h5".format(file.strip())
    data = h5py.File(filepath)
    image = data['image']
    test_list.append(image.shape[0])

print('>>>>>>>>>train<<<<<<<<<<<')
print('min:', min(train_list))
print('max:', max(train_list))
print('mean:', sum(train_list)/len(train_list))

print('>>>>>>>>>test<<<<<<<<<<<')
print('min:', min(test_list))
print('max:', max(test_list))
print('mean:', sum(test_list)/len(test_list))

print('>>>>>>>>>all<<<<<<<<<<<')
print('min:', min(train_list+test_list))
print('max:', max(train_list+test_list))
print('mean:', sum(train_list+test_list)/len(train_list+test_list))


