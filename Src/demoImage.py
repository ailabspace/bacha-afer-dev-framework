import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
from sklearn import preprocessing
import math
from keras.models import model_from_json
import csv 
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
model = model_from_json(open("../model/ANN_model1024.json", "r").read())
model.load_weights('../model/ANN_model1024.h5') #load weights best is Copy

#-----------------------------
emotions = ('Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt')
# initialize dlib's face detector and create a predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def euclidean(a, b):
    dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
    return dist 

# calculates distances between all 68 elements
def euclidean_all(a):  
    distances = ""
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            dist = euclidean(a[i], a[j])
            dist = "%.2f" % dist;
            distances = distances + " " + str(dist)
    return distances


csvName = "Test.csv"
        
file_exists = os.path.isfile(csvName)
    
if(file_exists == True):
    os.remove(csvName)
    
img = cv2.imread('../data/1_EeGMTlW4HL-ZAgPKnv1R8g.jpeg') #'data/CKPlus/Test_Original/2/S011_005_00000017.png'


bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

if len(bounding_boxes) > 0:
    
    for person in bounding_boxes:
        
        
        distances = []
        X , x_test = [] , []
        
        x , y, w,h = int(person[0]),int(person[1]), int(person[2]-person[0]), int(person[3]-person[1])
        
        cut =min(int(h * 20 / 100), int(h-w))
                
        h=h-cut
        y=y+cut
        #print (bounding_box)
#            roi = img[y:y+h, x-diff:x+w+diff]
        roi = img[y:y+h, x:x+w]
        roi1 = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        height, width, channels = roi1.shape
        x1=y1=0
        
        rect = dlib.rectangle(x1, y1, x1 + width, y1 + height)
        
        shape = predictor(roi1, rect)
        shape = face_utils.shape_to_np(shape)
        distances = euclidean_all(shape)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
    #    output = face_utils.visualize_facial_landmarks(roi1, shape)
        
        with open("Test.csv", 'a', encoding='utf-8-sig') as csvfile:
            fieldnames = ['vectors']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #                file_exists = os.path.isfile(csvName)
            #                if not file_exists:    
            #                    writer.writeheader()
            writer.writerow({'vectors': distances})

        
        if(len(distances)!=0):
            distances = distances.strip() 
            val = distances.split(" ")
            val = np.array(val)
            val = val.astype(np.float)
    #                val = np.expand_dims(val, axis = 1)  
            val = val.reshape(-1,1)
            minmax = preprocessing.MinMaxScaler()
            val = minmax.fit_transform(val)
            val = val.reshape(1,4624)

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