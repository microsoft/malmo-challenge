# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import sys
import numpy as np

OPENCV_AVAILABLE = False
PILLOW_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
    print('OpenCV found, setting as default backend.')
except ImportError:
    pass

try:
    import PIL

    PILLOW_AVAILABLE = True

    if not OPENCV_AVAILABLE:
        print('Pillow found, setting as default backend.')
except ImportError:
    pass


if not (OPENCV_AVAILABLE or PILLOW_AVAILABLE):
    raise ValueError('No image library backend found.'' Install either '
                     'OpenCV or Pillow to support image processing.')


def resize(img, shape):
    """
    Resize the specified image
    :param img: Image to reshape
    :param shape: New image shape
    :return:
    """
    if OPENCV_AVAILABLE:
        from cv2 import resize
        return resize(img, shape)
    elif PILLOW_AVAILABLE:
        from PIL import Image
        return np.array(Image.fromarray(img).resize(shape))


def rgb2gray(img):
    """
    Convert an RGB image to grayscale
    :param img: image to convert
    :return:
    """
    if OPENCV_AVAILABLE:
        from cv2 import cvtColor, COLOR_RGB2GRAY
        return cvtColor(img, COLOR_RGB2GRAY)
    elif PILLOW_AVAILABLE:
        from PIL import Image
        return np.array(Image.fromarray(img).convert('L'))
