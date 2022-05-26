import cv2
import numpy as np
import Knn
import time

'''
Detects the Colour of each pixel in runtime. 
1. Marks the ROI on the cam feed
2. crops the 9 pixels inside the ROI after converting it into HSV space
3. Extracts histogram of Hue and Saturation channels
4. Sends an array of peak histogram values of Hue and Saturation to Knn to detect colours
'''
Face_cordintes = np.array([[(180, 110), (240, 165)], [(270, 110), (330, 165)], [(360, 110), (420, 165)],
                           [(190, 195), (250, 250)], [(275, 195), (335, 250)], [(360, 195), (420, 250)],
                           [(195, 280), (250, 325)], [(280, 280), (330, 325)], [(360, 280), (410, 325)]])

Faces = np.array(([np.zeros((55, 60, 3)), np.zeros((55, 60, 3)), np.zeros((55, 60, 3)),
                   np.zeros((55, 60, 3)), np.zeros((55, 60, 3)), np.zeros((55, 60, 3)),
                   np.zeros((45, 55, 3)), np.zeros((45, 50, 3)), np.zeros((45, 50, 3))]), dtype=object)

cap = cv2.VideoCapture(0)


def detect_colour(Faces):
    feature_matrix = np.zeros((9, 2))
    for i, face in enumerate(Faces):
        hue = face[..., 0].flatten()
        sat = face[..., 1].flatten()
        hist_hue, hist_sat = np.histogram(hue, bins=np.arange(256)), np.histogram(sat, bins=np.arange(256))
        feature_matrix[i] = [np.argmax(hist_hue[0]), np.argmax(hist_sat[0])]
    print(feature_matrix)
    predicted_colours = Knn.predict_colour(feature_matrix)


while (True):
    _, frame = cap.read()
    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    i = 0
    for cordinates in Face_cordintes:
        point1 = tuple(cordinates[0])
        point2 = tuple(cordinates[1])
        cv2.rectangle(frame, point1, point2, (0, 0, 0), 2)

        x_len = abs(cordinates[0, 0] - cordinates[1, 0])
        y_len = abs(cordinates[0, 1] - cordinates[1, 1])

        Faces[i] = frame_copy[cordinates[0, 1]:cordinates[1, 1], cordinates[0, 0]:cordinates[1, 0]]
        i += 1

    detect_colour(Faces)
    cv2.imshow("img", frame)

    time.sleep(0.1)

    if (cv2.waitKey(1) == 27):
        break