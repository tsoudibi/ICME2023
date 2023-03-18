from generator import get_files, Img2ImgGenerator
import os
from sklearn.model_selection import train_test_split

def get_generators(batch_size, dataset_path,image_shape,real_world = False):
    x_train_files = []
    x_valid_files = []
    y_valid_files = []
    # y_train_files = get_files(dataset_path + '/drivable_maps/labels/train/*.png')
    # y_valid_files = get_files(dataset_path + '/drivable_maps/labels/val/*.png')
    if not real_world:
        y_train_files, y_valid_files = train_test_split(get_files(dataset_path + '/labels/class_labels/*.png'), test_size=0.0001)
    else:
        print(dataset_path + '/labels_real_world/*.png')
        y_train_files= get_files(dataset_path + '/labels_real_world/*.png')



    for file in y_train_files:
        if real_world==False:
            id = file.replace('\\', '/').split('/')[-1].split('_')[0]
            # x_file = dataset_path + '/images/100k/train/' + id + '.jpg'
            x_file = dataset_path + '/images/' + id + '.jpg'
        else:
            id = file.replace('\\', '/').split('/')[-1].split('.')[0]
            # x_file = dataset_path + '/images/100k/train/' + id + '.jpg'
            x_file = dataset_path + '/images_real_world/' + id + '.jpg'

        if not os.path.exists(x_file):
            print('Not exist ', x_file)
            y_train_files.remove(file)
            continue

        x_train_files.append(x_file)

    if len(y_valid_files)>0:
        for file in y_valid_files:
            if real_world==False:
                id = file.replace('\\', '/').split('/')[-1].split('_')[0]
                # x_file = dataset_path + '/images/100k/train/' + id + '.jpg'
                x_file = dataset_path + '/images/' + id + '.jpg'
            else:
                id = file.replace('\\', '/').split('/')[-1].split('.')[0]
                # x_file = dataset_path + '/images/100k/train/' + id + '.jpg'
                x_file = dataset_path + '/images_real_world/' + id + '.jpg'

            if not os.path.exists(x_file):
                print('Not exist ', x_file)
                y_valid_files.remove(file)
                continue

            x_valid_files.append(x_file)

    print('train: %d -> %d' % (len(x_train_files), len(y_train_files)))
    if len(y_valid_files)>0:
        print('valid: %d -> %d' % (len(x_valid_files), len(y_valid_files)))

    # original size : 720,1024
    # train_gen = Img2ImgGenerator(x_train_files, y_train_files, batch_size, x_shape=(288, 512,3), y_shape=(288, 512,3))
    # valid_gen = Img2ImgGenerator(x_valid_files, y_valid_files, batch_size, x_shape=(288, 512,3), y_shape=(288, 512,3))
    train_gen = Img2ImgGenerator(x_train_files, y_train_files, batch_size, x_shape=image_shape, y_shape=image_shape,real_world = real_world)
    if len(y_valid_files)>0:
        valid_gen = Img2ImgGenerator(x_valid_files, y_valid_files, batch_size, x_shape=image_shape, y_shape=image_shape,real_world = real_world)

    if len(y_valid_files)>0:
        return train_gen, valid_gen
    else:
        return train_gen

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, valid = get_generators(1, '/home/tsou/Desktop/ICME2023/iVS-ODSEG-Dataset')

    # for x, y in train:
    #     plt.imshow(x[0])
    #     plt.show()
    #     plt.imshow(y[0])
    #     plt.show()
    #     input()
