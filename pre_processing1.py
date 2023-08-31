from pdf2image import convert_from_path
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# TODO - change the path to the project path, There should be a datafolder in the hierarchy
project_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR'
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


# def skew_correction(img):
#     '''Taken from:
#      https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/'''
#
#     # grab the (x, y) coordinates of all pixel values that
#     # are greater than zero, then use these coordinates to
#     # compute a rotated bounding box that contains all
#     # coordinates
#     plt.figure(figsize=(20, 10))
#
#     img = img[:, :].copy()
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     # threshold the image, setting all foreground pixels to
#     # 255 and all background pixels to 0
#     thresh = cv2.threshold(img, 0, 255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     plt.imshow(thresh, cmap='gray')
#     plt.show()
#     coords = np.column_stack(np.where(thresh > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     # the `cv2.minAreaRect` function returns values in the
#     # range [-90, 0); as the rectangle rotates clockwise the
#     # returned angle trends to 0 -- in this special case we
#     # need to add 90 degrees to the angle
#     if angle < -45:
#         angle = -(90 + angle)
#     # otherwise, just take the inverse of the angle to make
#     # it positive
#     else:
#         angle = -angle
#
#     # rotate the image to deskew it
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h),
#                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#     # draw the correction angle on the image so we can validate it
#     cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     # show the output image
#     print("[INFO] angle: {:.3f}".format(angle))
#     cv2.imshow("Input", img)
#     cv2.imshow("Rotated", rotated)
#     cv2.waitKey(0)


def clean_image(image_path):
    """
    cleans the image and save it
    :param image_path:  the path of the image
    """
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{image_path}')
    img_copy = cv2.imread(fr'{image_path}', cv2.IMREAD_GRAYSCALE)
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
    #Text skew correction
    lines_detection(img,image_path)





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
    new_first_row = np.where(sum_of_pixels != 0)
    if not any(new_first_row[0]):
        # new_first_row is empty, finished extracting lines
        return
    first_row = np.where(sum_of_pixels != 0)[0][0]
    # get the index of the first row that the sum of the pixels is zero after the first row
    second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
    #     show the first line of the image
    line = img[first_row:second_row, :]
    # plt.imshow(line, cmap='gray')
    # plt.show()
    #     save the first line of the image
    plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
    #     crop the first line from the image
    img = img[second_row:, :]
    #     stage 2
    done_exracting_lines = False
    while (len(img) > 0) and (not done_exracting_lines):
        # get the sum of the pixels in each row
        sum_of_pixels = np.sum(img, axis=1)
        new_first_row = np.where(sum_of_pixels != 0)
        if not any(new_first_row[0]):
            # new_first_row is empty, finished extracting lines
            done_exracting_lines = True
            break
        # get the index of the next row that the sum of the pixels is not zer
        first_row = np.where(sum_of_pixels != 0)[0][0]
        # check if the sum is bigger than theshold
        # get the index of the next row that the sum of the pixels is zero after the first row
        second_row = np.where(sum_of_pixels[first_row:] == 0)[0][0] + first_row
        # show the next line of the image
        line = img[first_row:second_row, :]
        # plt.imshow(line, cmap='gray')
        # plt.show()
        line_dim = second_row - first_row
        # checks sufficient thershold or suffiecient line_dim  (number of rows included)
        if sum_of_pixels[first_row] > theshold or line_dim > 10:
            # save the next line of the image
            line_number += 1
            plt.imsave(fr'{image_line_segmentation_dir_path}\line_{line_number}.jpg', line, cmap='gray')
        # crop the next line from the image
        img = img[second_row:, :]


def word_segmentation(line_path, character_path, line_number):
    """
    segment the characters in the line. Each character was enclosed in a bounding box. To do this correctly,
    there needed to be some space between characters. Now the bounded image of character was further cropped from all
    four sides of the box. This was done by scanning for the first line with non-zero pixel value sum from top,
    bottom, left, and right of the bounding box :param line_path: the path of the line
    """
    # read image
    plt.figure(figsize=(20, 10))
    img = plt.imread(fr'{line_path}')
    #img = cv2.bitwise_not(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # RGB --> BGR
    img = img[:, :, ::-1].copy()
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    char_num = 1
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # character = character[y:y + h, x:x + w]
        # width, height = character.shape[:2]
        if (h * w) < 200 or h <= 8:
            #skip the current contour
            continue
        character = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        character = character[y:y + h, x:x + w]
        #perimeter = cv2.arcLength(contour,True)
        #plt.imshow(character, cmap='gray')
        #plt.show()
        # plt.imshow(character, cmap='gray')
        # plt.show()
        plt.imsave(fr'{character_path}\line_{line_number}_char_{char_num}.jpg', character, cmap='gray')
        char_num += 1

def lines_detection(image,image_path):
    # Read the image
    #img_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\questionnaires_jpg\empty_questionnair_FIRSTPAGEONLY\empty_questionnair_FIRSTPAGEONLY_0.jpg'
    #image = cv2.imread('img_path', 1)

    # # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a horizontal kernel for detecting lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    # Use morphologyEx to identify horizontal lines
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Find contours of the lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the detected lines and remove them
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # You can add conditions here to filter unwanted rectangles (too small, too large, etc.)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Filling the line with white color

    # Save or display the image
    #cv2.imshow('image', image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    plt.imsave(fr'{image_path}', image, cmap='gray')
    #cv2.waitKey(0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#
def main():
    convert_to_jpg_flag = True
    process_all_flag = True
    if convert_to_jpg_flag:
        convert_questionnairs_to_jpg()
    if process_all_flag:
        for questionnaire in os.listdir(questionnaires_jpg_dir_path):
            # creates separate folder to store the segmentations.
            parent_path = os.path.dirname(questionnaires_jpg_dir_path)
            segment_dir_path = parent_path + r'\Qs_segmentation'
            if not os.path.exists(segment_dir_path):
                os.mkdir(segment_dir_path)
            segment_questionnaire_path = fr'{segment_dir_path}\{questionnaire}'
            if not os.path.exists(segment_questionnaire_path):
                os.mkdir(segment_questionnaire_path)
            # check if folder "line_segmentation" exists
            if not os.path.exists(fr'{segment_questionnaire_path}\line_segmentation'):
                os.mkdir(fr'{segment_questionnaire_path}\line_segmentation')

            # access the directory where jpg questionnaires are stored.
            questionnaire_dir_path = fr'{questionnaires_jpg_dir_path}\{questionnaire}'

            for image in os.listdir(questionnaire_dir_path):
                # if os.path.isdir(fr'{questionnaire_dir_path}\{image}'):
                #     continue
                # check if line segmentation folder for this image exists
                image_line_segmentation_dir_path = fr'{segment_questionnaire_path}\line_segmentation\{image.split(".")[0]}'
                if not os.path.exists(image_line_segmentation_dir_path):
                    os.mkdir(image_line_segmentation_dir_path)

                image_path = fr'{questionnaire_dir_path}\{image}'

                clean_image(image_path)
                line_segmentation(image_path, image_line_segmentation_dir_path, theshold=2)
                #                character segmentation
                #                 check if character segmentation folder for this image exists
                image_character_segmentation_dir_path = fr'{segment_questionnaire_path}\character_segmentation'
                if not os.path.exists(image_character_segmentation_dir_path):
                    os.mkdir(image_character_segmentation_dir_path)
                character_path = fr'{image_character_segmentation_dir_path}\{image.split(".")[0]}'
                if not os.path.exists(character_path):
                    os.mkdir(character_path)
                line_number = 1
                for line in os.listdir(image_line_segmentation_dir_path):
                    # if line is a folder skip
                    if os.path.isdir(fr'{image_character_segmentation_dir_path}\{line}'):
                        continue
                    line_path = fr'{image_line_segmentation_dir_path}\{line}'
                    word_segmentation(line_path, character_path, line_number)
                    line_number += 1


main()
