import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
from sklearn import datasets
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from skimage.feature import hog


def image_prepare(argv):
    print("Entering image_prepare")

    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
 
    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
 
    # newImage.save("sample.png
 
    tv = list(newImage.getdata())  # get pixel values
 
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    # print(tva)
    return tva

def visualize_784(x_list):
    print("Entering visualize_784")

    newArr=[[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j]=x_list[0][k]
            k=k+1

    return newArr

def HOG_Compute(img_array):
    print("Entering HOG_Compute")
    im2arr = np.array(img_array).reshape(1,28,28)
    gray = np.array(im2arr, dtype='uint8')
    #hog = cv2.HOGDescriptor((28,28),(10,10),(3,3),(10,10),4)
    hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (14, 14),12)
    x_new = np.zeros((gray.shape[0],108))
    for i in range(gray.shape[0]):
        x_new[i] = hog.compute(gray[i]).reshape(1,-1)
    return x_new

def read_image_into_array(imageFile):
    print("read_image_into_array")
    tva = image_prepare(imageFile)
    imageArray = visualize_784([tva])
    x_new_hog = HOG_Compute(imageArray)
    return x_new_hog

def build_gaussian_model():
    print("Entering build_gaussian_model")
    dataset = keras.datasets.mnist

    (x_raw_train,y_train),(x_raw_test,y_test) = dataset.load_data()

    x_raw_train = np.concatenate([x_raw_train, x_raw_test])
    y_train = np.concatenate([y_train, y_test])
    hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (14, 14),12)
    xtrain = np.zeros((x_raw_train.shape[0],108))
    xtest = np.zeros((x_raw_test.shape[0],108))   

    for i in range(x_raw_train.shape[0]):
        xtrain[i] = hog.compute(x_raw_train[i]).reshape(1,-1)

    for i in range(x_raw_test.shape[0]):
        xtest[i] = hog.compute(x_raw_test[i]).reshape(1,-1)

    _,prob_y = np.unique(y_train, return_counts=True)
    prob_y = prob_y/len(y_train)
    num_classes = len(prob_y)

    (means, stdev) = train_naive_bayes(num_classes, xtrain, y_train)

    print("prob_y = {}, num_classes = {}, means = {}, stdev = {}".format(prob_y, num_classes, means, stdev))
    return (prob_y, num_classes, means, stdev)

def train_naive_bayes(num_classes, xtrain, y_train):
    print("Entering train_naive_bayes")
    means = np.zeros((num_classes, xtrain.shape[1]),dtype=np.float64)
    stdevs = np.zeros((num_classes, xtrain.shape[1]),dtype=np.float64)
    for ind,iv in enumerate(np.unique(y_train)):
        indices = np.where(iv == y_train)
        means[ind] = np.mean(xtrain[indices],axis=0)
        stdevs[ind] = np.std(xtrain[indices], axis=0)

    return (means, stdevs)

def predict(imageArray, prob_y, means, stdevs, num_classes):
    print("Entering predict")
    return np.argmax([np.log(prob_y[k]) + log_gaussian(imageArray.reshape(1, -1), 
                      means[k], stdevs[k]) for k in range(num_classes)])

def log_gaussian(X, mu, sigma):
    # print("Enterig log_gaussian")
    return -(np.sum(np.log(sigma)) + 0.5 * np.sum(((X - mu) / sigma) ** 2)).reshape(-1, 1)

def main():
    imageFile = "Downloads/009_1.png"
    imageArray = read_image_into_array(imageFile)
    (prob_y, num_classes, means, stdev) = build_gaussian_model()

    predictedValue = predict(imageArray, prob_y, means, stdev, num_classes)

    print("Image file: {}".format(imageFile))
    print("Predicted value: ", predictedValue)

main()
