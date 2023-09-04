import cv2
import numpy as np
import os

from matplotlib import pyplot as plt


def word_segmentation(line_path):
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{line_path}')
    img = cv2.bitwise_not(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    # RGB --> BGR
    img = img[:, :, ::-1].copy()
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(np.shape(img))
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    plt.imshow(img, cmap='gray')
    plt.show()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    word_num = 1
    filtered_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # character = character[y:y + h, x:x + w]tryouts.py
        # width, height = character.shape[:2]
        if (h * w) < 200 or (h < np.shape(img)[0]):
            continue
        character = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        character = character[y:y + h, x:x + w]
        filtered_contours.append(contour)
        plt.imshow(character, cmap='gray')
        plt.show()
    # Display the image with bounding rectangles
    cv2.imshow('Bounding Rectangles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_removal(img_path, var):
    if var == 1:
        # Load the image
        image = cv2.imread(fr'{img_path}', cv2.IMREAD_GRAYSCALE)

        # Apply thresholding to segment text from background
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

        # Find connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        # Remove lines based on their aspect ratio
        line_removed_image = image.copy()
        img_height = np.shape(line_removed_image)[0]
        img_width = np.shape(line_removed_image)[1]
        for label in range(1, n_labels):  # Skip background label 0
            x, y, width, height, area = stats[label]
            #component_mask = (labels == label).astype(np.uint8)  # Create a mask for the current component
            #length = np.sum(component_mask == 1)  # Calculate the length by summing up all pixel values equal to 1
            # Remove components that have a higher width than height (lines)
            if (height < 5 and width > 100 ):
                line_removed_image[y:y + height, x:x + width] = 255  # Replace with background color

        # Display the result
        cv2.imshow('Line Removed Image', line_removed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if var == 2:
        plt.figure(figsize=(20, 10))
        image = cv2.imread(fr'{img_path}', cv2.IMREAD_GRAYSCALE)
        # plt.imshow(image, cmap='gray')
        # plt.show()
        # Create a copy of the original image to draw lines on
        image_with_lines = image.copy()
        img_copy = image.copy()
        # Apply Gaussian blur to reduce noise and improve line detection
        blurred_image = cv2.GaussianBlur(img_copy, (5, 5), 0)
        plt.imshow(blurred_image, cmap='gray')
        plt.show()
        # Apply Canny edge detection to find edges
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

        # Find lines using the Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

        # Draw detected lines on the copy of the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Iterate through the detected lines and remove them from the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5)

        # Display the original image with detected lines
        plt.imshow(image_with_lines, cmap='gray')
        plt.show()
        cv2.imshow('Image with Lines', image_with_lines)

        # Display the image with lines removed
        plt.imshow(image, cmap='gray')
        plt.show()
        cv2.imshow('Image with Lines Removed', image)
        # save the image
        #plt.imsave(fr'{image_path}', image, cmap='gray')


def main():
    # # get the index of the first row that the sum of the pixels is not zero
    # arr = np.array([[0, 0], [0, 0]])
    # check = np.sum(arr, axis=1)
    # #print(check)
    # whr = np.where(check != 0)
    # print(whr)
    # # Check if whr is empty using any()
    # if any(whr[0]):
    #     print("whr is not empty.")
    # else:
    #     print("whr is empty.")

    dir_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\questionnaires_jpg\empty_questionnair'
    data_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\Qs_segmentation\empty_questionnair\line_segmentation\empty_questionnair_0'

    line_removal(fr'{dir_path}\empty_questionnair_0.jpg', var=2)

    # line_number = 1
    # for line in os.listdir(data_path):
    #     # if line is a folder skip
    #     if os.path.isdir(fr'{data_path}\{line}'):
    #         continue
    #     line_path = fr'{data_path}\{line}'
    #     word_segmentation(line_path)
    #     line_number += 1


main()
