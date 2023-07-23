import pydicom
from PIL import Image
import cv2
import numpy as np
import imutils

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

## 이미지 크기 변형이 필요할 듯함 - 큰 이미지를 사용할 수록 연산속도가 느려짐
## 1)이미지 자르기 
## 2)