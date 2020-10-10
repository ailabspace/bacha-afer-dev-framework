# Bacha's AFER Dev Framework

**Dependencies:**

OpenCV
DLIB
Tensorflow
Keras
CSV
imutils
numpy
scikit-learn 
matplotlib
pickle

A development framework for Automatic Facial Expression Recognition (AFER) systems base on DNN and CNN models. This baseline development framework includes:

1.	Source code for image preprocessing, generate 68 facial landmarks, calculate distance vectors among every facial point and to save cropped images, landmark images and generate CSV file. You can run following files to do these tasks:
generate_csv_cropimages_LMimages_Train_MTCNN.py (For Training Set)
generate_csv_cropimages_LMimages_MTCNN.py (For Testing Set)

2.	Source code to build, train, evaluate and save the baseline DNN and CNN models on the data provided in the CSV files and generate graphs for loss and confusion matrix.
NN_Reshaped.py (For DNN model training)
NN_CNN.py (For CNN model training)

3.	Source code to apply the trained models on unseen images and videos including live video input.
demo.py (For video or webcam facial expressions Recognition through trained DNN model)
demoImage.py (For single image based facial expressions Recognition through trained DNN model)
demo_CNN.py (For video or webcam facial expressions Recognition through trained CNN model)
demoImage_CNN.py (For single image based facial expressions Recognition through trained CNN model)

Dataset Distribution:
![Alt text](results/ANN/Untitled.png?raw=true)

**Results:**

DNN
![Alt text](results/ANN/Normalized ANN.png?raw=true)

DNN
![Alt text](results/CNN/Normalized CNN.png?raw=true)

...