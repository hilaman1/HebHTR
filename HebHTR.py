from processFunctions import *
from predictWord import *
import os
import matplotlib.pyplot as plt


class HebHTR:

    def __init__(self, img_path):
        self.img_path = img_path
        self.original_img = plt.imread(fr'{img_path}', cv2.IMREAD_GRAYSCALE)
        # RGB --> BGR
        self.original_img = self.original_img[:, :, ::-1].copy()

    def imgToWord(self, iterations=5, decoder_type='best_path'):
        transcribed_words = []
        model = getModel(decoder_type=decoder_type)
        transcribed_words.extend(predictWord(self.original_img, model))
        return transcribed_words
