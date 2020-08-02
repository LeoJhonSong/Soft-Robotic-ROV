# test what exactly the key you pressed is in OpenCV

import cv2
import numpy

img = numpy.zeros((100, 100))
cv2.imshow('key test', img)

while True:
    res = cv2.waitKey(0)
    print('You pressed %d (0x%x), LSB: %d (%s)' % (res, res, res % 256, repr(chr(res%256)) if res%256 < 128 else '?'))
    print(res)