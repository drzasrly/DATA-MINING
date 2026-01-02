import cv2
import numpy as np

def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def resize(img, size=(256,256)):
    return cv2.resize(img, size)

def add_title(img, text):
    img = to_bgr(img)
    overlay = img.copy()

    cv2.rectangle(overlay, (0,0), (img.shape[1], 20), (0,0,0), -1)
    cv2.putText(
        overlay, text, (5,15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0,255,0), 1
    )
    return overlay

def make_canvas(rows):
    canvas_rows = []
    for row in rows:
        row_imgs = [resize(add_title(img, title))
                    for img, title in row]
        canvas_rows.append(np.hstack(row_imgs))
    return np.vstack(canvas_rows)
