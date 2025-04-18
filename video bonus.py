import cv2
import numpy as np

# Load the query image (object to detect)
query_img = cv2.imread('query.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(query_img, None)

# Use FLANN based matcher for better performance in video
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Open video file or webcam
cap = cv2.VideoCapture(0)  # Change to "video.mp4" for a file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    target_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    if des2 is not None:
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Compute homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = query_img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw bounding box
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
