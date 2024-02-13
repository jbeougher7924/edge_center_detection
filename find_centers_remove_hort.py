import cv2
import numpy as np

def find_centers_from_threshold(thresh, min_area, max_area):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for contour in contours:
        # Calculate area of each contour
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Calculate centroid of each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
    return centers

# Read the original image
original_image = cv2.imread('boat-1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, thresh_image = cv2.threshold(original_image, 150, 255, cv2.THRESH_BINARY)

# Define minimum and maximum area thresholds
min_area_threshold = 250  # Adjust as needed
max_area_threshold = 5000  # Adjust as needed

# Find centers of boats from thresholded image
centers = find_centers_from_threshold(thresh_image, min_area_threshold, max_area_threshold)

# Convert the thresholded image to BGR (to be able to draw red circles)
thresh_image_bgr = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)

# Draw circles on thresholded image
for center in centers:
    cv2.circle(thresh_image_bgr, center, 5, (0, 0, 255), -1)

# Save the modified image to file
cv2.imwrite('boat_centers_max250.png', thresh_image_bgr)

# Display the results
cv2.imshow('Boat Centers', thresh_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
