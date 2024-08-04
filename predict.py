import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('covid_model.h5') # load the model here
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
#classifierLoad = tensorflow.keras.models.load_model('model_tamil.h5')
test_image1 = cv2.imread('1 (2).jpeg',0)
#test_image1=cv2.VideoCapture(0, cv2.CAP_DSHOW)
#result, images = test_image1.read()
'''if result:
    
    
    cv2.imshow("input image ",images)
    
   
    cv2.imwrite("1 (1).jpeg", images)

else:
    print(' Move to this part is input image has some error')'''

test_image = image.load_img('1 (2).jpeg', target_size = (200,200))  # load the sample image here
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)
if result[0][0] == 1:
    print("covid")
    test_image1 = cv2.resize(test_image1, (200,200))                
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)
    
elif result[0][1] == 1:
    print("normal")
    test_image1 = cv2.resize(test_image1, (200,200))                
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)
   
elif result[0][2] == 1:
    print("ம்")
    test_image1 = cv2.resize(test_image1, (200,200))                
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)

elif result[0][3] == 1:
    print("மா")
    test_image1 = cv2.resize(test_image1, (200,200))                
    cv2.imshow('sampleimage',test_image1)
    cv2.waitKey(0)
