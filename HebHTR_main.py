from HebHTR import *

# Create new HebHTR object.
img_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\Qs_segmentation\Test_HebHTR-MoDel\character_segmentation\Test_HebHTR-MoDel_0\line_3_char_1.jpg'
#img_path = r'C:\Users\Gal\Source\Repos\NLP\HebHTR\data\Qs_segmentation\Test_HebHTR-MoDel\line_segmentation\Test_HebHTR-MoDel_0\line_3.jpg'

img = HebHTR(img_path)

# Infer words from image.

text = img.imgToWord(iterations=5, decoder_type='word_beam')
