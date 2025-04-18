import cv2
import numpy as np
from matplotlib import pyplot as plt

query_img = cv2.imread(r'C:\Users\SAM\Downloads\Mini Assignment\query.jpg', cv2.IMREAD_GRAYSCALE)
target_img = cv2.imread(r'C:\Users\SAM\Downloads\Mini Assignment\target.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()


kp1, des1 = sift.detectAndCompute(query_img, None)
kp2, des2 = sift.detectAndCompute(target_img, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)


result = cv2.drawMatches(query_img, kp1, target_img, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(14, 7))
plt.imshow(result)
plt.title("SIFT Matching")
plt.axis('off')
plt.show()
