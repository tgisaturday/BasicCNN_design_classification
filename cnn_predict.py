import time
import numpy as np
import pandas as pd
import random
import logging
import json
import sys
import os
import tensorflow as tf
import data_helper
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

def predict_unseen_data():
    """Step 0: load trained model and parameters"""
    params = json.loads(open('./parameters.json').read())
    checkpoint_dir = sys.argv[1]
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    test_file = "./dataset/"
    columns = ['section', 'class', 'subclass', 'description']
    selected = ['section', 'description']

    test_list = []
    for path, dirs, files in os.walk(test_file):
        if files:
            for filename in files:
                fullname = os.path.join(path,filename)
                test_list.append(fullname)
                
    data = []
    print("Listing all datas in testset.")
    start = time.time()
    global counter_image
    global total_dataset
    total_dataset = len(test_list)
    counter_image = 0
    for filename in test_list:
        data.append([filename.split('/')[2], filename.split('/')[3], filename.split('/')[4], clean_image(filename)])
    df = pd.DataFrame(data, columns=columns)
    print("\nExecution time = {0:.5f}".format(time.time() - start))

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = json.loads(open('./labels.json').read())
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    

    total_dataset = len(test_list)
    #x_raw = [example['abstract'] for example in test_examples]
    #x_test = [data_helper.clean_str(x) for x in x_raw]
    x_test = df[selected[1]].tolist()
    y_test = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    
    logging.info('The number of x_test: {}'.format(len(x_test)))
    logging.info('The number of y_test: {}'.format(len(y_test)))

    x_test = np.array(x_test)

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        y_test = np.argmax(y_test, axis=1)
        correct_predictions = sum(all_predictions == y_test)
        logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
        logging.critical('The prediction is complete')

    labels = json.loads(open('labels.json').read())
    predicted_labels = []
    for i in range(len(all_predictions)):
        temp = labels[int(all_predictions[i])]
        predicted_labels.append(temp)
    answer = df['section'].tolist()
    df['predicted'] = predicted_labels
    df['answer'] = answer
    columns = ['answer','predicted','name']
    df.to_csv('predictions_all.csv', index=False, columns=columns, sep=',', encoding='utf-8')
if __name__ == '__main__':
    # python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
    predict_unseen_data()
