import pydicom
from PIL import Image
import cv2
import numpy as np
import imutils
from rembg import remove

# threshold (임계값줘서 큰건 키우고, .. )
# --> 문턱 값 이상이면 어떤 값으로 바꾸어주고 낮으면 0으로 바꾸어주는 기능
'''
1. 이미지 처리 : threshold를 통해서 x-ray로 찍은 이미지 선명(하얀 부분은 하얗게 ~)하게 해주기
2. 이미지 크기가 3072 x 3072 이므로 이를 처리할 수 있는 인공지능 모델 찾아서 해보기(나중에 사진만 넣을 수 있게)
'''

# Load the DICOM file
ds = pydicom.dcmread("../dcm/test1.dcm")

# Convert DICOM pixel data to a PIL image object
pil_img = Image.fromarray(ds.pixel_array)

# Save as JPEG
pil_img.save("../output/output.tif")
src = cv2.imread("../output/output.tif", cv2.IMREAD_GRAYSCALE)

# threshold
_, thr = cv2.threshold(src, 28, 255, cv2.THRESH_BINARY)
cv2.imwrite("../output/thr.tif", thr)

# 이미지 회전
rotated = imutils.rotate(thr, 180)
cv2.imwrite("../output/rotate.tif", rotated)

# 이미지 좌우반전
flipped = cv2.flip(rotated, 1)
cv2.imwrite("../output/flip.tif", flipped)

# 배경제거
rm_background = remove(flipped)
cv2.imwrite("../output/back_img.tif", rm_background)

# 이미지 자르기
cut_img = rm_background[860:2750, 610:2500]
cv2.imwrite("../output/cut_img.tif", cut_img)

# 크기 조절
resize_img = cv2.resize(cut_img, (224, 224))
cv2.imwrite("../output/resize_img.tif", resize_img)