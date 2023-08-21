import pydicom
from PIL import Image
import cv2
import numpy as np
import imutils
from rembg import remove

# DCM to TIF
ds = pydicom.dcmread("../dcm/test1.dcm")
pil_img = Image.fromarray(ds.pixel_array)
pil_img.save("../output/output.tif")
src = cv2.imread("../output/output.tif", cv2.IMREAD_GRAYSCALE)

# image_processing
_, thr = cv2.threshold(src, 28, 255, cv2.THRESH_BINARY)

rotated = imutils.rotate(thr, 180)
flipped = cv2.flip(rotated, 1)
rm_background = remove(flipped)
cut_img = rm_background[860:2750, 610:2500]


resize_img = cv2.resize(cut_img, (224, 224))

# (224 x 244) image 
cv2.imwrite("../output/resize_img.tif", resize_img)