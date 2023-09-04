# from process_functions import *
import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

'''
Class tis:

Parameters:
    - img_path (string): A path to text-based image to work on.

Returns:
    A new text_segmentation object.
'''


class Tis:

    def __init__(self, img_path):
        self.img_path = img_path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        final = cv2.bitwise_not(img)
        self.img = final

    '''
    Function crop_text:

    Parameters:
        - output_dir (string): A path to save the image to. If None is given as
            a parameter, images will be saved in a directory called 
            "text_segmentation" inside the original image's parent directory.
        - iterations (int): Number of dilation iterations that will be done on
        the image. Default value is set to 5.
        - dilate (bool): Whether to dilate the text in the image or not.
            Default is set to 'True'. It is recommended to dilate the image for
            better segmentation. 


    Returns:
        None.
        Saves images of all words from the text in the output path.
    '''

    def crop_text(self, output_dir, iterations=5, dilate=True):
        sharp_img = self.__sharpen_text(self.img)
        conts = self.__contours(sharp_img, iterations, dilate)

        if not output_dir:
            print("Please insert output directory.")
            return

        self.__crop_words(sharp_img, conts, output_dir)


    '''
    Function draw_rectangles:

    Parameters:
        - output_path (string): A path to save the image to. If None is given as
            a parameter, image will be saved in the original image parent
            directory.
        - iterations (int): Number of dilation iterations that will be done on
        the image. Default value is set to 5.
        - dilate (bool): Whether to dilate the text in the image or not.
            Default is set to 'True'. It is recommended to dilate the image for
            better segmentation. 

    Returns:
        None.
        Saves the image in the output path.
    '''

    def draw_rectangles(self, output_path=None, iterations=5, dilate=True):
        #sharp_img = self.__sharpen_text(self.img)
        #conts = self.__contours(sharp_img, iterations, dilate)
        conts = self.__contours(self.img, iterations, dilate)

        if not output_path:
            parent_path = os.path.dirname(self.img_path)
            output_path = parent_path + 'draw_rectangles_result.png'

        self.__crop_characters(self.img, conts, str(output_path))
        for i in conts:
            self.__draw_rects(self.img, i, str(output_path))




    # Clean image background and sharpen text.
    def __sharpen_text(self, img):
        # read image
        plt.figure(figsize=(20, 10))
        #thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # show the image
        plt.imshow(thresh, cmap='gray')
        plt.show()
        return thresh

    # Find contours of words in the text.
    def __contours(self, img, iterations=3, dilate=False):
        # Dilate image for better segmentation
        if dilate:
            im = self.__dilate_img(img, iterations)
            contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Dilate image for better segmentation in contours detection.
    def __dilate_img(self, img, iterations):
        # Clean all noises.
        denoised = cv2.fastNlMeansDenoising(img, None, 50, 7, 21)
        # Negative the image.
        imagem = cv2.bitwise_not(denoised)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(imagem, kernel, iterations=iterations)

        # Negative it again to original color
        final = cv2.bitwise_not(dilate)

        return final

    # Crop the words the contours function found.
    def __crop_words(self, img, conts, output_dir):
        counter = 0

        for i in conts:
            x, y, w, h = cv2.boundingRect(i)
            cropped = img[y:y + h, x:x + w]
            width, height = cropped.shape[:2]

            # Avoid cropping very small contours e.g dots, commas.
            if width * height > 10:
                plt.imshow(cropped, cmap='gray')
                plt.show()
                cv2.imwrite(str(output_dir) + str(counter) + ".png", cropped)
                plt.imsave('output_path', img, cmap='gray')

            counter = counter + 1

        return

    def __crop_characters(self, img, conts, output_dir):
        counter = 0

        for i in conts:
            x, y, w, h = cv2.boundingRect(i)
            cropped = img[y:y + h, x:x + w]
            width, height = cropped.shape[:2]

            # Avoid cropping very small contours e.g dots, commas.
            if width * height > 20:
                cv2.imwrite(str(output_dir) + str(counter) + "char.png", cropped)

            counter = counter + 1

    # Draw rectangles around the contours function found.
    def __draw_rects(self, img, contour, path_save):
        (x, y, w, h) = cv2.boundingRect(contour)

        # Clean all small contours out.
        if (h * w) < 50:
            return
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 60), 1)
        plt.imshow(img, cmap='gray')
        plt.show()



def main():
    # Creates a new image object.
    path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\line_10_croped.jpg'
    output_dir = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\outputsTis\cropped_words'
    output_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\outputsTis\rec.png'
    img = Tis(path)

    # Crops words from text and saves them to 'cropped_words' directory.
    img.crop_text(output_dir, iterations=5, dilate=True)

    # Draws rectangles around all words in the text, and saves the result to
    # 'result_rec.png'.
    img.draw_rectangles(output_path, iterations=1)
    plt.imsave('output_path', img, cmap='gray')
    plt.imshow(img)
    plt.show()


main()
