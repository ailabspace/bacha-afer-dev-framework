# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob
import os
import csv 
import math
import shutil
#from mtcnn.mtcnn import MTCNN

import tensorflow as tf
tf.reset_default_graph()
import detect_face
from keras import backend as K


sess = tf.Session()

pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.8 ]  # three steps's threshold
factor = 0.709 # scale factor

# initialize dlib's face detector and create a predictor
#detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# calculate distance vectors between each point and the center

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#detector = MTCNN()

def detect_parts(image, filename, path2):
    distances = []
    shape = []
    # resize the image, and convert it to grayscale
    #image = imutils.resize(image, width=200, height=200)
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
#    rects = detector(gray, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width, channels = image.shape
    x=y=0
    
    rect = dlib.rectangle(x, y, x + width, y + height)
    # loop over the face detections
    
    shape = predictor(image, rect)
    shape1 = face_utils.shape_to_np(shape)
    distances = euclidean_all(shape1)
    shape = np.array_repr(shape1).replace('\n ', '')
    shape = shape.replace(',      ', ' ')
    shape = shape.replace(',', '')
    shape = shape.replace('  ', ' ')
    shape = shape.replace('[ ', '[')
    shape = shape.replace('] ', ']')
    shape = shape.replace('array(', '')
    shape = shape.replace(')', '')
        # output = face_utils.visualize_facial_landmarks(image, shape)
        # visualize all facial landmarks with a transparent overlay
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)    
    if distances != []:
        newpath1 = path2.replace('_Original','_MTCNN_Crop')
        if not os.path.exists(os.path.dirname(newpath1)):
            os.makedirs(os.path.dirname(newpath1))
        cv2.imwrite(newpath1, gray)
        
        newpath2 = path2.replace('_Original','_MTCNN_LM')
        newpath3 = newpath2.replace('.png','')
        
        get_landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        
        for idx, point in enumerate(get_landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(image, str(idx), pos,                
                            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.4,
                            color=(0, 0, 255))
            cv2.circle(image, pos, 3, color=(0, 255, 255))
        
        output = face_utils.visualize_facial_landmarks(image, shape1)
        
        if not os.path.exists(os.path.dirname(newpath2)):
            os.makedirs(os.path.dirname(newpath2))
        cv2.imwrite(newpath2, output)
        np.save(newpath3, shape1)
        
        #print(np.load(newpath3+'.npy'))
        
    return distances, shape

def euclidean(a, b):
    dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
    return dist 

def euclidean_all(a):  # calculates distances between all 68 elements
    distances = ""
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            dist = euclidean(a[i], a[j])
            dist = "%.2f" % dist;
            distances = distances + " " + str(dist)
    return distances



def generate_csv(dirName, csvName):
    file_exists = os.path.isfile(csvName)
    
    if(file_exists == True):
        os.remove(csvName)
    
    if(os.path.exists(LM_directory)):
        print(LM_directory)
        shutil.rmtree(LM_directory)
        
    if(os.path.exists(Crop_directory)):
        print(Crop_directory)
        shutil.rmtree(Crop_directory)
        
    for path in glob.glob(dirName): # assuming png
        img = cv2.imread(path)
        path2 = path
        path, filename = os.path.split(path)
        path, label = os.path.split(path)
        
        height, width = img.shape[:2]
        maxlength = max(height, width)
        minsize = max(int(maxlength / 10), 40)
#        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # print(faces) #locations of detected faces

        if len(bounding_boxes) > 0:
#            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
            for person in bounding_boxes:
                
                distances = []
#                print(person)
                
    #            bounding_box = result[0]['box']
    #            keypoints = result[0]['keypoints']
                x , y, w,h = int(person[0]),int(person[1]), int(person[2]-person[0]), int(person[3]-person[1])
                
    #            cut =int(h - w)
    #            print(cut)
                cut =min(int(h * 20 / 100), int(h-w))
                
                h=h-cut
                y=y+cut
#                diff = int((h-w)/2)
                #print (bounding_box)
    #            roi = img[y:y+h, x-diff:x+w+diff]
                roi = img[y:y+h, x:x+w]
                roi1 = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)
                
                distances, shape = detect_parts(roi1, filename, path2)
                if distances != []:
                    distances = np.asarray(distances)
                
                    with open(csvName, 'a', encoding='utf-8-sig', newline='') as csvfile:
                        fieldnames = ['filename','emotion', 'landmarks', 'vectors']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        file_exists = os.path.isfile(csvName)
                        if not file_exists:    
                            writer.writeheader()
                        writer.writerow({'filename': filename , 'emotion': label, 'landmarks': shape, 'vectors': distances})
                        #print (label, filename)
                break
    #            cv2.imshow('img',img)
    #            cv2.waitKey(1)
        

# ---------------------------------------------------------------------------------------------------


testing_csvName = "CKUBD_Test_DLIB_MTCNN.csv"
testing_dirName = '../data/Test_Original/*/*'
LM_directory = '../data/Test_MTCNN_LM/'
Crop_directory = '../data/Test_MTCNN_Crop/'
generate_csv(testing_dirName, testing_csvName)

print("Success: CSV files are generated!")

sess.close()
K.clear_session()

# ---------------------------------------------------------------------------------------------------








