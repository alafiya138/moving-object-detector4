import cv2

# Load an image from file
image = cv2.imread('input.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny
edges = cv2.Canny(gray, 100, 200)

# Save the result
cv2.imwrite('edges.jpg', edges)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()