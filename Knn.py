import numpy as np
import math
import cv2
from scipy import stats
from collections import Counter

'''This script Loads the Training Dataset and then performs k-Nearest-Neighbouring algorithm on the provided dataset.
the Colour_detection.py sends the features (max hue and sat from theor respective histograms) of the 9 pixels of the cube
'''

fh = open("./Dataset/Complete_data.txt", 'r')
data = np.array([])
line = fh.readlines()
data = np.append(data, line)
np.random.shuffle(data)
fh.close()

X0 = np.ones(len(data))
X = np.ones((len(data), 2))
X1 = np.array([], dtype=int)
X2 = np.array([])
Y = np.array([])
# print(data[0].split(',')[0])

for line in data:
    X1 = np.append(X1, int(line.split(',')[0]))
    X2 = np.append(X2, int(line.split(',')[1]))
    Y = np.append(Y, line.split(',')[2])

X[..., 0] = X1
X[..., 1] = X2

def hashmap(color):
  return {
    'o': "\033[7;31;43m橙\033[0m",
    'w': "\033[1;30;47m白\033[0m",
    'p': "\033[1;10;45m粉\033[0m",
    'y': "\033[1;30;43m黄\033[0m",
    'g': "\033[1;40;42m绿\033[0m",
    'b': "\033[1;40;44m蓝\033[0m"
  }[color]

def predict_colour(feature_Mat):
    final_prediction_seq = np.empty(9, dtype=str)
    for i, feature in enumerate(feature_Mat):
        final_prediction_seq[i] = str (predict_single_face_colour(feature))
    color_list = list(map(hashmap, final_prediction_seq))
    color_list = list(map(hashmap, final_prediction_seq))
    for i in range(9):
        print(color_list[i], end='  ')
        if (i+1) % 3 == 0:
            print('\n')

def predict_single_face_colour(features):
    closest_dist = np.zeros((540))
    closest_label = np.empty(540, dtype=object)
    for i, training in enumerate(X):
        Htrain, Strain = training[0], training[1]
        closest_dist[i] = math.sqrt(math.pow((features[0] - Htrain), 2) + math.pow(features[1] - Strain, 2))
        closest_label[i] = Y[i]
    u = np.transpose(np.array([closest_dist, closest_label]))
    sorted = u[u[:, 0].argsort()]

    colour = stats.mode(sorted[1:20, 1])
    final_colour = str(colour[0])
    return final_colour[2]


def create_features():
    img = cv2.imread("60.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = img[..., 0].flatten(), img[..., 1].flatten()
    hist_hue, hist_sat = np.histogram(h, bins=np.arange(256)), np.histogram(s, np.arange(256))
    hue, sat = np.argmax(hist_hue[0]), np.argmax(hist_sat[0])
    predict_colour(np.array([hue, sat]))

# create_features()