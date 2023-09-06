import cv2
import numpy as np

def main(inputImage):
        
    # Read the image
    #image = cv2.imread(inputImage)
    nparr = np.frombuffer(inputImage, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output_image = image.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operations to remove noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area to get the biggest two
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:2]

    # Get X coordinates of the biggest red areas
    x_coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_coordinates.extend(range(x, x+w))

    # Initialize an empty list to hold the line midpoints
    midpoints = []
    lines = []

    # Loop from top to bottom every 5 pixels
    for y in range(0, mask.shape[0], 10):
        row = mask[y, :]
        indices = np.where(row == 255)[0]

        if len(indices) > 1:
            for i in range(len(indices) - 1):
                start_gap = indices[i]
                end_gap = indices[i + 1]

                if end_gap - start_gap > 1:
                    if start_gap in x_coordinates and end_gap in x_coordinates:
                        midpoint = (start_gap + end_gap) // 2
                        midpoints.append(midpoint)
                        lines.append(((start_gap, y), (end_gap, y)))

    # Cluster midpoints based on their X coordinate
    clusters = {}
    threshold = 10

    for midpoint, line in zip(midpoints, lines):
        added_to_cluster = False
        for key in clusters.keys():
            if abs(midpoint - key) <= threshold:
                clusters[key].append(line)
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters[midpoint] = [line]

    # Find the largest cluster based on the number of lines
    largest_cluster_key = max(clusters.keys(), key=lambda x: len(clusters[x]))

    # Draw only the lines from the largest cluster
    for start, end in clusters[largest_cluster_key]:
        cv2.line(output_image, start, end, (0, 255, 255), 2)

    # Initialize an empty string to store the output data
    output_string = ""

    # Loop through the lines in the largest cluster to fill the output string
    for start, end in clusters[largest_cluster_key]:
        y_coordinate = start[1]
        width_of_gap = end[0] - start[0]
        output_string += "[{},{}] ".format(y_coordinate, width_of_gap)

    print("Output String:", output_string) 
    return output_string


