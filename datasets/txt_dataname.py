import os

data_root = 'dataset/CD/ChangeDetectionDataset/Real/subset'

train_txt = 'train.txt'
test_txt = 'test.txt'
val_txt = 'val.txt'

subsets = ['train', 'test', 'val']
t = 'A'


for subset in subsets:
    image_name_list = os.listdir(os.path.join(data_root, subset,t))
    for image in image_name_list:
        with open(subset+'.txt', 'a') as f:
            f.write(image+'\n')
            print(image)