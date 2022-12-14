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

DNN Model:

![Alt text](results/ANN/Normalized ANN.png?raw=true)

CNN Model:

![Alt text](results/CNN/Normalized CNN.png?raw=true)

**Demo:**

[![Watch the video](https://img.youtube.com/vi/WRs_PJl_4bA/0.jpg)](https://youtu.be/WRs_PJl_4bA)

**Publication**

Rehman B., Ong W.H., Ngo T.D. (2021) A Development Framework for Automated Facial Expression Recognition Systems. In: Suhaili W.S.H., Siau N.Z., Omar S., Phon-Amuaisuk S. (eds) Computational Intelligence in Information Systems. CIIS 2021. Advances in Intelligent Systems and Computing, vol 1321. Springer, Cham. https://doi.org/10.1007/978-3-030-68133-3_16

**Project Page**

https://ailab.space/projects/multimodal-human-intention-perception/
