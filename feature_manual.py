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

import numpy as np

def compute_glcm(image, distance=1, angle=0, levels=256):
    glcm = np.zeros((levels, levels), dtype=np.float32)
    rows, cols = image.shape

    dx = int(np.round(distance * np.cos(angle)))
    dy = int(np.round(distance * np.sin(angle)))

    for i in range(rows - dy):
        for j in range(cols - dx):
            i_val = image[i, j]
            j_val = image[i + dy, j + dx]
            glcm[i_val, j_val] += 1

    return glcm
