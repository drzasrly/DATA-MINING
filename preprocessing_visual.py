import cv2
import numpy as np
from config import HSV_LOWER, HSV_UPPER, IMG_SIZE


def preprocess_visual(image, show=True):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)


    mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
    mask_median = cv2.medianBlur(mask, 5)
    segmented = cv2.bitwise_and(image, image, mask=mask_median)


    if show:
        cv2.imshow("Original", image)
        cv2.imshow("Hue", h)
        cv2.imshow("Saturation", s)
        cv2.imshow("Value", v)
        cv2.imshow("Mask", mask)
        cv2.imshow("Mask Median", mask_median)
        cv2.imshow("Segmented", segmented)

    return segmented, mask_median