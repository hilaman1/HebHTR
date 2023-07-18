from pdf2image import convert_from_path
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


# TODO - change the path to the project path
project_path = r'C:\Users\hillu\OneDrive\מסמכים\nlp\final_project'
data_path = fr'{project_path}\data'
questionnaires_pdf_dir_path = fr'{data_path}\questionnaires_pdf'
questionnaires_jpg_dir_path = fr'{data_path}\questionnaires_jpg'


def pdf_to_image(pdf_path, jpg_path, pdf_name):
    """
    convert pdf to images and save them in the jpg_path folder
    :param pdf_path:  the path of the pdf file
    :param jpg_path:  the path of the jpg folder
    :param pdf_name:  the name of the pdf file
    """
    images = convert_from_path(pdf_path)
    for i in range(len(images)):
        images[i].save(fr'{jpg_path}\{pdf_name}_{i}.jpg', 'JPEG')


def convert_questionnairs_to_jpg():
    for pdf_file in os.listdir(questionnaires_pdf_dir_path):
        pdf_path = fr'{questionnaires_pdf_dir_path}\{pdf_file}'
        pdf_name = pdf_path.split('\\')[-1].split('.')[0]
        # check if there is a folder with the name of the pdf
        jpg_path = fr'{questionnaires_jpg_dir_path}\{pdf_name}'
        if not os.path.exists(fr'{jpg_path}'):
            os.mkdir(fr'{jpg_path}')
        # convert the pdf to images and save them in the folder
        pdf_to_image(pdf_path, jpg_path, pdf_name)


def clean_image(image_path):
    """
    clean the image and save it
    :param image_path:  the path of the image
    """
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{image_path}')
    # RGB --> BGR
    img = img[:, :, ::-1].copy()
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # clean the image
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    # save the image
    plt.imsave(fr'{image_path}', img, cmap='gray')
    # show the image
    plt.imshow(img, cmap='gray')
    plt.show()


def line_segmentation(image_path, image_line_segmentation_dir_path, theshold):
    """
        Starting from the top of the image, the row of pixels where the sum of pixel values was not zero was searched.
         This marked the beginning of the first line in the document.
        The row where the sum of pixel values was zero after the first line marked the bottom of the current line.
    :param image_path: the path of the image
    """
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{image_path}')
    # RGB --> BGR
    img = img[:, :, ::-1].copy()
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     convert 255 to 0 and 0 to 1
    img = np.where(img == 255, 0, 1)

    line_number = 1
#     stage 1
    # get the sum of the pixels in each row
    sum_of_pixels = np.sum(img, axis=1)
    # get the index of the first row that the sum of the pixels is not zero
    first_row = np.where(sum_of_pixels != 0)[0][0]
    # get the index of the first row that the sum of the pixels is zero after the first row
    second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
#     show the first line of the image
    line = img[first_row:second_row, :]
    plt.imshow(line, cmap='gray')
    plt.show()
#     save the first line of the image
    plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
#     crop the first line from the image
    img = img[second_row:, :]
#     stage 2
    while len(img) > 0:
        line_number += 1
        # get the sum of the pixels in each row
        sum_of_pixels = np.sum(img, axis=1)
        # get the index of the next row that the sum of the pixels is not zero
        first_row = np.where(sum_of_pixels != 0)[0][0]
        # check if the sum is bigger than theshold
        # get the index of the next row that the sum of the pixels is zero after the first row
        second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
        # show the next line of the image
        line = img[first_row:second_row, :]
        plt.imshow(line, cmap='gray')
        plt.show()
        if sum_of_pixels[first_row] > theshold:
            # save the next line of the image
            plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
        # crop the next line from the image
        img = img[second_row:, :]


def character_segmentation(line_path, character_path, line_number):
    """
    segment the characters in the line. Each character was enclosed in a bounding box. To do this correctly,
    there needed to be some space between characters. Now the bounded image of character was further cropped from all
    four sides of the box. This was done by scanning for the first line with non-zero pixel value sum from top,
    bottom, left, and right of the bounding box :param line_path: the path of the line
    """
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{line_path}')
    # RGB --> BGR
    img = img[:, :, ::-1].copy()
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    char_num = 1
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (h * w) < 200:
            continue
        character = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        character = character[y:y + h, x:x + w]
        plt.imshow(character, cmap='gray')
        plt.show()
        plt.imsave(fr'{character_path}\line_{line_number}_char_{char_num}.jpg', character, cmap='gray')
        char_num += 1








def main():
    convert_to_jpg_flag = True
    process_all_flag = True
    if convert_to_jpg_flag:
        convert_questionnairs_to_jpg()
    if process_all_flag:
        for questionnaire in os.listdir(questionnaires_jpg_dir_path):
            questionnaire_dir_path = fr'{questionnaires_jpg_dir_path}\{questionnaire}'
            # check if folder "line_segmentation" exists
            if not os.path.exists(fr'{questionnaire_dir_path}\line_segmentation'):
                os.mkdir(fr'{questionnaire_dir_path}\line_segmentation')
            for image in os.listdir(questionnaire_dir_path):
                # check if line segmentation folder for this image exists
                image_line_segmentation_dir_path = fr'{questionnaire_dir_path}\line_segmentation\{image.split(".")[0]}'
                if not os.path.exists(image_line_segmentation_dir_path):
                    os.mkdir(image_line_segmentation_dir_path)
                image_path = fr'{questionnaire_dir_path}\{image}'
                # clean_image(image_path)
                # line_segmentation(image_path, image_line_segmentation_dir_path, theshold=2)
#                character segmentation
#                 check if character segmentation folder for this image exists
                character_path = fr'{image_line_segmentation_dir_path}\character_segmentation'
                if not os.path.exists(character_path):
                    os.mkdir(character_path)
                line_number = 1
                for line in os.listdir(image_line_segmentation_dir_path):
                    # if line is a folder skip
                    if os.path.isdir(fr'{image_line_segmentation_dir_path}\{line}'):
                        continue
                    line_path = fr'{image_line_segmentation_dir_path}\{line}'
                    character_segmentation(line_path, character_path, line_number)
                    line_number += 1



main()