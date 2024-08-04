# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os,glob

def detect_and_predict_mask(frame, maskNet):
    (h, w) = frame.shape[:2]
    locs = []
    preds = []
    patches = []
    # Define the window size
    windowsize_r = 50
    windowsize_c = 50
    
    # Crop out the window and calculate the histogram
    for r in range(0,frame.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,frame.shape[1] - windowsize_c, windowsize_c):
            
            this_block = frame[r:r+windowsize_r,c:c+windowsize_c,:]    
            this_block = cv2.cvtColor(this_block, cv2.COLOR_BGR2RGB)
            this_block = cv2.resize(this_block, (200, 200))
            this_block = img_to_array(this_block)
            this_block = preprocess_input(this_block)
            patches.append(this_block)
        
            locs.append((c, r, c+windowsize_c, r+windowsize_r))
    if len(patches) > 0:
        patches = np.array(patches, dtype="float32")
        preds = maskNet.predict(patches, batch_size=32)
    return (locs, preds)

# load the face mask detector model from disk
maskNet = load_model("model_tamil.h5")

vs = cv2.VideoCapture()

#vs = cv2.VideoCapture('http://192.168.43.1:8080/video')
CATEGORIES =["A", "E","M","Ma"]


# loop over the frames from the video stream
while True:
    ret,orig_frame = vs.read()

# images_list = glob.glob('./test_data/*')
# # images_list = ['reg_fakebuy4get1_scanned (52).jpg' , 'reg_orig_iphone(8).jpg']
# for this_image in images_list:
    
    start_time = time.time()
    # orig_frame = cv2.imread(this_image)
    (h, w) = orig_frame.shape[:2]
    # 	frame = imutils.resize(frame, width=400)
    orig_frame_cropped = orig_frame[:,:int(w/10),:]
    (locs, preds) = detect_and_predict_mask(orig_frame_cropped, maskNet)
    
    label_detected = []
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        pred = list(pred) 
        index_letter = pred.index(max(pred))
        label = CATEGORIES[index_letter]
        # (mask, withoutMask) = pred
        # Status = mask > withoutMask
        # label = "Scanned_Copy" if mask > withoutMask else "Not Scanned"
        # label_detected.append(0 if mask > withoutMask else 1)
        # color = (0, 255, 0) if label == "Scanned_Copy" else (0, 0, 255)
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # cv2.putText(orig_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # cv2.rectangle(orig_frame, (startX, startY), (endX, endY), color, 2)
        
        # show the output frame
    color = (0, 0, 255) if label_detected.count(1) > label_detected.count(0) else (0, 255, 0)

    # label = "Not Scanned" if label_detected.count(1) > label_detected.count(0) else "Scanned_Copy"
    cv2.putText(orig_frame, label, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    end_time  = time.time()
    # print ('Time Taken:',end_time - start_time )
    
    cv2.imshow("Frame", orig_frame)
    key = cv2.waitKey(1) & 0xFF
    print ('Image Given is: ',label)

cv2.destroyAllWindows()
# vs.stop()
