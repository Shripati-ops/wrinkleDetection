import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

im_path = "/Volumes/Personal/PythonProjects/comicBookAnalysis/images/1b.jpg"

img = cv2.imread(im_path)
  
# Convert the img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_ridges(gray, sigma=1.0):
  H_elems = hessian_matrix(gray, sigma=sigma, use_gaussian_derivatives=True)
  maxima_ridge, minima_ridge = hessian_matrix_eigvals(H_elems)
  return maxima_ridge, minima_ridge

a, b = detect_ridges(gray, sigma=1.0)
a = cv2.normalize(a, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
a[a<np.mean(a)] = 0
a[a!=0]=255

k = 7
kernel = np.zeros((k, k) ,np.uint8)
kernel[k//2] = 1

a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)

mask = np.zeros_like(a)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=a, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
th_ = 15
for cnt in contours:
    if(cv2.contourArea(cnt)>th_):
        cv2.drawContours(image=mask, contours=cnt, contourIdx=-1, color=255, thickness=2, lineType=cv2.LINE_AA)
                
img[mask>0] = (0, 0, 255)
result = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (255,255,255), 5)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img)
cv2.waitKey(0)