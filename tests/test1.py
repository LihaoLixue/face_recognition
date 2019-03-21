from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from face_recognition import api
# img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
lena = Image.open(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
print(lena.mode)
print(lena.getpixel((0,0)))
lena.show()
