import time
import numpy as np
import pandas as pd
import random
import sys
import os


import cv2
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_triangle
from skimage.filters import try_all_threshold
from skimage.transform import resize

from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage import restoration


def clean_image(img_file):
    """Clean sentence"""
    global counter_image
    global total_dataset
    tmp_img = cv2.imread(img_file, 0)
    tmp_img = cv2.medianBlur(tmp_img, 3)
    tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 0)

    image = np.asarray(tmp_img)

   #image = restoration.denoise_tv_chambolle(image, weight=0.1)
    #thresh = threshold_yen(image)
    thresh = threshold_triangle(image)

    bw = closing(image > thresh, square(2))
    cleared = bw.copy()
    contour = np.logical_xor(image, cleared)
    width = image.shape[1];
    height = image.shape[0];
    if width > height:
        new_width = 250
        new_height = round(250 * height / width)
        if new_height == 0:
            new_height = 2
        if new_height % 2 != 0:
            new_height += 1

    elif width < height:
        new_width = round(250 * width / height)
        new_height = 250
        if new_width == 0:
            new_width = 2
        if new_width % 2 != 0:
            new_width += 1
    else:
        new_width = 250
        new_height = 250
                
    processed = resize(contour, (new_height,new_width), mode='edge')
    processed = np.pad(processed, ((int((256-new_height)/2), int((256-new_height)/2)),
                                                   (int((256-new_width)/2), int((256-new_width)/2))), mode='edge')
    #plt.imshow(processed)
    #plt.show()
    counter_image += 1
    sys.stdout.write("\r Parsed: %d / %d" %(counter_image, total_dataset))
    sys.stdout.flush()
    return processed


def load_data_and_labels(foldername):
    """Load sentences and labels"""
    columns = ['section', 'class', 'subclass', 'abstract']
    selected = ['section', 'abstract']
    global counter_image
    global total_dataset
    file_list = []
    for path, dirs, files in os.walk(foldername):
        if files:
            for filename in files:
                fullname = os.path.join(path, filename)
                file_list.append(fullname)

    random.shuffle(file_list)

    data = []
    print("Listing all datas in dataset.")
    start = time.time()
    counter_image = 0
    total_dataset = len(file_list)
    for filename in file_list:
        data.append([filename.split('/')[2], filename.split('/')[3], filename.split('/')[4], clean_image(filename)])
    df = pd.DataFrame(data, columns=columns)
    print("\nExecution time = {0:.5f}".format(time.time() - start))

    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    x_raw = df[selected[1]].tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    dataset = './dataset/'
    load_data_and_labels(dataset)
