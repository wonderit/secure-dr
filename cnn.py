# https://www.kaggle.com/code/siddhantsinghrawat/diabetic-retinopathy-analysis-using-cnn/notebook
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import sys
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import scikitplot as skplt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-el", "--epoch_limit", help="Set epoch limit", type=int, default=80)
parser.add_argument("-li", "--log_interval", help="Set batch interval for log", type=int, default=5)
parser.add_argument("-b", "--batch_size", help="Set batch size for log", type=int, default=32)
parser.add_argument("-v", "--valid_size", help="Set validation size for log", type=float, default=0.1)
parser.add_argument("-n", "--n_splits", help="Set validation size for log", type=int, default=3)
parser.add_argument("-c", "--is_comet", help="Set isTest", action='store_true')
parser.add_argument("-p", "--comet_project", help="Set project name", type=str, default='secure-dr-plaintext')
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-m", "--model_name", help="model name(alex, lenet, resnet, vgg)", type=str, default='alexnet')
parser.add_argument("-w", "--image_width", help="image width, height", type=int, default=32)

args = parser.parse_args()

result_folder_name = 'result'
if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

# Importing Data
dataset19 = pd.read_csv('./data/labels/trainLabels19.csv')
print(dataset19)

# check model


# Visualizing Data
names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']
print(dataset19['diagnosis'].value_counts())
ax = sns.barplot(x=names, y=dataset19.diagnosis.value_counts().sort_index())
fig = ax.get_figure()
fig.savefig('data_count19.png')

# Importing Data 2015
dataset15 = pd.read_csv('./data/labels/trainLabels15.csv')
dataset15.columns = ['id_code', 'diagnosis']
print(dataset15)

print(dataset15['diagnosis'].value_counts())
ax = sns.barplot(x=names, y=dataset15.diagnosis.value_counts().sort_index())
fig = ax.get_figure()
fig.savefig('data_count15.png')

# Balancing Data
# Now we will take 900 images in total for each class. So to complete the 900 images we will take
# the majority of images from 'dataset' and if necessary take the rest of the required images from 'dataset15'

# index  Final_Img_count   Image taken from dataset 1
# 0          900                   (0)
# 1          900                 (530)
# 2          900                   (0)
# 3          900                 (707)
# 4          900                 (605)
level_0 = dataset19[dataset19.diagnosis == 0].sample(n=900)
level_2 = dataset19[dataset19.diagnosis == 2].sample(n=900)

level_1 = dataset15[dataset15.diagnosis == 1].sample(n=530)
level_3 = dataset15[dataset15.diagnosis == 3].sample(n=707)
level_4 = dataset15[dataset15.diagnosis == 4].sample(n=605)

dataset19 = dataset19[dataset19['diagnosis'] > 0]
dataset19 = dataset19[dataset19['diagnosis'] != 2]
print(dataset19['diagnosis'].value_counts())

dataset19 = pd.concat([level_0, level_2, dataset19])
dataset19 = dataset19.sample(frac=1)
print(dataset19['diagnosis'].value_counts())

dataset15 = pd.concat([level_1, level_3, level_4])
dataset15 = dataset15.sample(frac=1)

print(dataset15['diagnosis'].value_counts())

# IMPORTING SELECTED IMAGES FROM THE DATASET
# RESIZING THE IMPORTING DATA
images = []
for i, image_id in enumerate(tqdm(dataset19.id_code)):
    im = cv2.imread(f'./data/resized train 19/{image_id}.jpg')
    im = cv2.resize(im, (args.image_width, args.image_width))
    images.append(im)

for i, image_id in enumerate(tqdm(dataset15.id_code)):
    im = cv2.imread(f'./data/resized train 15/{image_id}.jpg')
    im = cv2.resize(im, (args.image_width, args.image_width))
    images.append(im)

# PREPROCESSING OF IMAGE DATA
# random image from imported data
plt.imsave('data_example.png', images[-30])


# APPLYING GAUSSIAN BLUR NOISE FILTER
# This function will act as a filter for the image data
def load_colorfilter(image, sigmaX=10):
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, args.image_width)
    return image


for i in range(len(images)):
    output = load_colorfilter(images[i])
    images[i] = output

# image after filtering
plt.imsave('data_example_gbf.png', images[-30])

images = np.array(images)
print('shape of images: ', images.shape)

# VISUALIZING BALANCED DATASET
dataset = pd.concat([dataset19, dataset15])
print(dataset['diagnosis'].value_counts())

ax = sns.barplot(x=names, y=dataset.diagnosis.value_counts().sort_index())
fig = ax.get_figure()
fig.savefig('balenced_data_count.png')

# SCALING/NORMALISING IMAGE DATASET
X = images / 255.0
y = dataset.diagnosis.values

# Cleaning some RAM memory space
del images, level_1, level_3, level_4, level_0, dataset19

# Image Augmentation
# Applying image augmentation
sys.stdout.flush()
aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, \
                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, \
                         horizontal_flip=True, fill_mode="nearest")

# SPLITTING OF DATASET IN TRAIN AND TEST DATA
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1)#
# X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#######################################################################################

X, X_test, y, y_test = train_test_split(X, y, test_size=args.valid_size, stratify=y)
X.shape, X_test.shape, y.shape, y_test.shape


# Function defined to plot the curves during training

def display_training_curves(training, validation, title, subplot, fold):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    plt.savefig(f'./{result_folder_name}/training_curves_{title}_{fold}.png')


# TRAINING OF MODEL
# - DESIGNING THE CONVOLUTIONAL NEURAL NETWORK MODEL
# - USING STRATIFIED K-FOLD CROSS VALIDATION TECHNIQUE TO SPLIT THE TRAINING DATA INTO TRAINING AND VALIDATION SETS
# - COMPILE AND TRAIN THE MODEL FOR EACH SPLIT
# - PLOT THE TRAINING CURVES FOR EACH SPLIT

BS = args.batch_size  # Batch size
accuracy = []

############ USING STRATIFIED K-FOLD CROSS VALIDATION TECHNIQUE ##########

skf = StratifiedKFold(n_splits=args.n_splits)
skf.get_n_splits(X, y)

fold_no = 1

for train, test in skf.split(X, y):
    if fold_no > args.n_splits - 2:
        continue
    # Design of CNN Model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(args.image_width, args.image_width, 3), activation='relu', strides=(1, 1), padding="valid"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding="valid"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding="valid"),
        tf.keras.layers.MaxPooling2D(2, 2),
        #
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding="valid"),
        # tf.keras.layers.MaxPooling2D(2, 2),
        #
        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding="valid"),
        # tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    if fold_no == 1:
        print('model', model)

    # Compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])

    # Training
    history = model.fit_generator(aug.flow(X[train], y[train], batch_size=BS),
                                  validation_data=(X[test], y[test]),
                                  epochs=args.epoch_limit, verbose=1)

    # Evaluate score
    acc = model.evaluate(X[test], y[test])
    accuracy.append(acc[1])

    # Plotting training curves
    display_training_curves(
        history.history['loss'],
        history.history['val_loss'],
        'loss', 211, fold_no)

    display_training_curves(
        history.history['acc'],
        history.history['val_acc'],
        'accuracy', 212, fold_no)
    # Increase fold number
    fold_no = fold_no + 1

# serialize model to JSON
model_json = model.to_json()
with open(f"./{result_folder_name}/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f"./{result_folder_name}/model.h5")
print("Saved model to disk")
model.save(f"./{result_folder_name}/modelsave.h5")
print("Saved modelsave to disk")

# we can see the minimum and maximum validation accuracy received after training on the training dataset
print(accuracy)

# thus we can assume the mean accuracy of the model on the training set to be:
a = sum(accuracy) / len(accuracy)
print(f'Mean evaluated accuracy of model : {a}')

# MODEL LAYER DIAGRAM
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

tf.keras.utils.plot_model(model, to_file=f'./{result_folder_name}/model.png')

# ANALYSIS OF TRAINING MODEL
# predicting training labels
# y_train_pred = model.predict_classes(X)
y_train_pred = model.predict(X)
y_train_pred = y_train_pred.argmax(axis=-1)

# Accuracy of train prediction
print('\nAccuracy of training data prediction : {:.2f}\n'.format(accuracy_score(y, y_train_pred)))

# confusion matrix for training set
confusion = confusion_matrix(y, y_train_pred)
print('Confusion Matrix of training data prediction \n')
print(confusion)

# Visualizing confusion matrix for train data
skplt.metrics.plot_confusion_matrix(y, y_train_pred, figsize=(8, 8))
plt.savefig(f'./{result_folder_name}/confusion_matrix_train.png')

# Classification report
print('\nClassification Report of training set : \n')
print(classification_report(y, y_train_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']))

# PREDICTING TEST RESULTS
# Accuracy of test prediction
# y_pred = model.predict_classes(X_test)
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

# Confusion matrix of the test data
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

# Visualizing confusion matrix for test data
skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(8, 8))
plt.savefig(f'./{result_folder_name}/confusion_matrix_test.png')

# Classification report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']))
