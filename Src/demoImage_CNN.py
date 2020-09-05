import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
from sklearn import preprocessing
import math
from keras.models import model_from_json
import os

import tensorflow as tf
tf.reset_default_graph()
import detect_face
from keras import backend as K


sess = tf.Session()

pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.8 ]  # three steps's threshold

factor = 0.709 # scale factor
#-----------------------------
#face expression recognizer initialization
# Using pretrained model
model = model_from_json(open("../model/CNN_model.json", "r").read())
model.load_weights('../model/CNN_model.h5') #load weights best is Copy
#model.save_weights('./model/reducedweightedmode512.h5')
#-----------------------------

emotions = ('Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt')
# initialize dlib's face detector and create a predictor

csvName = "Test.csv"
        
file_exists = os.path.isfile(csvName)
    
if(file_exists == True):
    os.remove(csvName)
    
img = cv2.imread('../data/1_EeGMTlW4HL-ZAgPKnv1R8g.jpeg') #'data/CKPlus/Test_Original/4/S073_002_00000024.png'


bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)


if len(bounding_boxes) > 0:
    
    for person in bounding_boxes:
        
        #            keypoints = result[0]['keypoints']
        x , y, w,h = int(person[0]),int(person[1]), int(person[2]-person[0]), int(person[3]-person[1])
        
        cut =min(int(h * 20 / 100), int(h-w))
                
        h=h-cut
        y=y+cut
        #print (bounding_box)
#            roi = img[y:y+h, x-diff:x+w+diff]
        roi = img[y:y+h, x:x+w]
        
        #            cv2.rectangle(img,(x-diff,y),(x+w+diff,y+h),(255,0,0),2) #draw rectangle to main image
        roi1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi1 = cv2.resize(roi1, (48, 48))
  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
 
        val = np.asarray(roi1) 
        val = val.reshape(1,48,48)
        val = val.reshape(1,48,48,1)
        val = val.astype('float32')
        

        
        predictions = model.predict(val) #store probabilities of 6 expressions
        #find max indexed array ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
        # 'Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt'
        print ("Neutral: %", predictions[0][0]/1.0 * 100)
        print ("Angry: %", predictions[0][1]/1.0 * 100)
        print ("Disgust: %", predictions[0][2]/1.0 * 100)
        print ("Fear: %", predictions[0][3]/1.0 * 100)
        print ("Happy: %", predictions[0][4]/1.0 * 100)
        print ("Sad: %", predictions[0][5]/1.0 * 100)    
        print ("Surprised: %", predictions[0][6]/1.0 * 100)     
        print ("Contempt: %", predictions[0][7]/1.0 * 100)     
        print ("----------------------"    )    
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        
        #write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    cv2.imwrite('img.png',img)
        
        
sess.close()
K.clear_session()
