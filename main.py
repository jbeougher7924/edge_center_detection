import cv2
import numpy as np

# Read the IR image
image = cv2.imread('boat-1.jpg', cv2.IMREAD_GRAYSCALE)

# Method 1: Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Method 2: Laplacian Edge Detection
laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)

# Method 3: Canny Edge Detection
canny_edges = cv2.Canny(image, 100, 200)

# Method 4: Thresholding
_, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

# Method 5: Contour Detection
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Sobel Edges', sobel_edges.astype(np.uint8))
cv2.imshow('Laplacian Edges', laplacian_edges.astype(np.uint8))
cv2.imshow('Canny Edges', canny_edges)
cv2.imshow('Thresholding', thresh)
cv2.imshow('Contour Detection', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()