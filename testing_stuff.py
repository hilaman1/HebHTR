from pdf2image import convert_from_path
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# TODO - change the path to the project path
project_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR'
data_path = fr'{project_path}\data'
questionnaires_pdf_dir_path = fr'{data_path}\questionnaires_pdf'
questionnaires_jpg_dir_path = fr'{data_path}\questionnaires_jpg'

#
# def pdf_to_image(pdf_path, jpg_path, pdf_name):
#     """
#     convert pdf to images and save them in the jpg_path folder
#     :param pdf_path:  the path of the pdf file
#     :param jpg_path:  the path of the jpg folder
#     :param pdf_name:  the name of the pdf file
#     """
#     images = convert_from_path(pdf_path)
#     for i in range(len(images)):
#         images[i].save(fr'{jpg_path}\{pdf_name}_{i}.jpg', 'JPEG')
#
#
# def convert_questionnairs_to_jpg():
#     for pdf_file in os.listdir(questionnaires_pdf_dir_path):
#         pdf_path = fr'{questionnaires_pdf_dir_path}\{pdf_file}'
#         pdf_name = pdf_path.split('\\')[-1].split('.')[0]
#         # check if there is a folder with the name of the pdf
#         jpg_path = fr'{questionnaires_jpg_dir_path}\{pdf_name}'
#         if not os.path.exists(fr'{jpg_path}'):
#             os.mkdir(fr'{jpg_path}')
#         # convert the pdf to images and save them in the folder
#         pdf_to_image(pdf_path, jpg_path, pdf_name)
#
# def clean_image(image_path):
#     """
#     cleans the image and save it
#     :param image_path:  the path of the image
#     """
#     # read image
#     plt.figure(figsize=(20, 10))
#     img = plt.imread(fr'{image_path}')
#     # RGB --> BGR
#     img = img[:, :, ::-1].copy()
#     # convert to gray scale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # convert to binary
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     # clean the image
#     kernel = np.ones((2, 2), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
#     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#     img = cv2.medianBlur(img, 3)
#     # save the image
#     plt.imsave(fr'{image_path}', img, cmap='gray')
#     # show the image
#     plt.imshow(img, cmap='gray')
#     plt.show()
#
#
# def line_segmentation(image_path, image_line_segmentation_dir_path, theshold):
#     """
#         Starting from the top of the image, the row of pixels where the sum of pixel values was not zero was searched.
#          This marked the beginning of the first line in the document.
#         The row where the sum of pixel values was zero after the first line marked the bottom of the current line.
#     :param image_path: the path of the image
#     """
#     # read image
#     plt.figure(figsize=(20, 10))
#     img = plt.imread(fr'{image_path}')
#     # RGB --> BGR
#     img = img[:, :, ::-1].copy()
#     # convert to gray scale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # convert to binary
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# #     convert 255 to 0 and 0 to 1
#     img = np.where(img == 255, 0, 1)
#
#     line_number = 1
# #     stage 1
#     # get the sum of the pixels in each row
#     sum_of_pixels = np.sum(img, axis=1)
#     print(sum_of_pixels)
#     # get the index of the first row that the sum of the pixels is not zero
#     print(img.shape)
#
#     first_row = np.where(sum_of_pixels != 0)[0][0]
#     # get the index of the first row that the sum of the pixels is zero after the first row
#     second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
# #     show the first line of the image
#     line = img[first_row:second_row, :]
#     plt.imshow(line, cmap='gray')
#     plt.show()
# #     save the first line of the image
#     plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
# #     crop the first line from the image
#     img = img[second_row:, :]
# #     stage 2
#     while len(img) > 0:
#         # get the sum of the pixels in each row
#         sum_of_pixels = np.sum(img, axis=1)
#         new_first_row = np.where(sum_of_pixels != 0)
#         if (not new_first_row[0].any()):
#             # new_first_row is empty, finished extracting lines
#             break
#         # get the index of the next row that the sum of the pixels is not zero
#         line_number += 1
#         first_row = np.where(sum_of_pixels != 0)[0][0]
#         # check if the sum is bigger than theshold
#         # get the index of the next row that the sum of the pixels is zero after the first row
#         second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
#         # show the next line of the image
#         line = img[first_row:second_row, :]
#         plt.imshow(line, cmap='gray')
#         plt.show()
#         if sum_of_pixels[first_row] > theshold:
#             # save the next line of the image
#             plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
#         # crop the next line from the image
#         img = img[second_row:, :]
#
#
#

# Read the image
img_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\questionnaires_jpg\empty_questionnair_FIRSTPAGEONLY\empty_questionnair_FIRSTPAGEONLY_0.jpg'

# Read the image
image = cv2.imread(img_path, 1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Create a kernel for horizontal line detection
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

# Use morphological operations to detect horizontal lines
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Dilate the original thresholded text image to create anchor points
anchor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_text = cv2.dilate(thresh, anchor_kernel, iterations=1)

# Create anchor points only where the lines touch the text
anchors = cv2.bitwise_and(detected_lines, dilated_text)

# Subtract the anchor points from the detected lines
line_wo_anchors = detected_lines - anchors

# Remove lines that don't touch text from the original image
result = cv2.add(image, cv2.cvtColor(line_wo_anchors, cv2.COLOR_GRAY2BGR))

# Show the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def main():
#     convert_to_jpg_flag = True
#     process_all_flag = True
#     if convert_to_jpg_flag:
#         convert_questionnairs_to_jpg()
#     questionnaire_dir_path = fr'{questionnaires_jpg_dir_path}\empty_questionnair_first_page'
#     for image in os.listdir(questionnaire_dir_path):  # for each image (page) of the file
#     # check if line segmentation folder for this image exists
#         image_line_segmentation_dir_path = fr'{questionnaire_dir_path}\line_segmentation\{image.split(".")[0]}'
#         if not os.path.exists(image_line_segmentation_dir_path):
#             os.mkdir(image_line_segmentation_dir_path)
#         image_path = fr'{questionnaire_dir_path}\{image}'
#         clean_image(image_path)
#         line_segmentation(image_path, image_line_segmentation_dir_path, theshold=2)



#main()