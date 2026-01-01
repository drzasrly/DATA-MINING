import cv2
import numpy as np

LEVELS = 16
IMG_SIZE = 256

def color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)]


def glcm_features(gray):
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = (gray / (256 / LEVELS)).astype(int)
    glcm = np.zeros((LEVELS, LEVELS))

    for i in range(gray.shape[0] - 1):
        for j in range(gray.shape[1] - 1):
            glcm[gray[i, j], gray[i, j + 1]] += 1


    glcm /= glcm.sum() + 1e-8
    i, j = np.indices(glcm.shape)


    contrast = np.sum(glcm * (i - j) ** 2)
    energy = np.sum(glcm ** 2)
    homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
    correlation = np.sum((i * j) * glcm)

    return [contrast, energy, homogeneity, correlation]

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return color_features(image) + glcm_features(gray)