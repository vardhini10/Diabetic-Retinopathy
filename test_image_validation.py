from image_validator import validate_uploaded_image
from PIL import Image
import numpy as np
import cv2

img=np.zeros((300,300,3),dtype=np.uint8)+50
cv2.circle(img,(150,150),80,(150,150,150),-1)
Image.fromarray(img).save('temp_test_eye.png')
valid,image,msg=validate_uploaded_image('temp_test_eye.png')
print('valid',valid,'msg',msg)
