import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
import dlib
import cv2
from sklearn import preprocessing
from imutils import face_utils
import numpy as np
from keras.optimizers import SGD

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
from keras.layers.normalization import BatchNormalization
import os
import glob
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import pickle


num_classes = 8 # 'Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', Surprise', 'Contempt'
batch_size = 64
epochs = 100
fit = True
seed(8)
img_rows, img_cols = 48, 48

Train_directory = '../data/Train_MTCNN_Crop/*/*'
Test_directory = '../data/Test_MTCNN_Crop/*/*'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=16)


    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    
# --------------------------------------------------------------------
# Data preperation from csv file
def prepare_data(Train_directory, Test_directory):
    X, Y = [], []
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for path in glob.glob(Train_directory): # assuming png
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        path, filename = os.path.split(path)
        path, emotion = os.path.split(path)
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
        X.append(img)
        Y.append(emotion)
    
    x_train = X
    y_train = Y    
    
    X, Y, testfilename = [], [], []   
    
    for path in glob.glob(Test_directory): # assuming png
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        path, filename = os.path.split(path)
        path, emotion = os.path.split(path)
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
        X.append(img)
        Y.append(emotion)
        testfilename.append(filename)
    
    x_test = X
    y_test = Y    
    
    
    
    print ("********Training set size: ", str(len(x_train)))
    print ("********Testing set size: ", str(len(x_test)))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Removed code
#    minmax = preprocessing.MinMaxScaler()
#    x_train = minmax.fit_transform(x_train)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Removed code
    # Normalization of the testing data   
#    minmax = preprocessing.MinMaxScaler()
#    x_test = minmax.fit_transform(x_test)

    return x_train, y_train, x_test, y_test, testfilename

# --------------------------------------------------------------------
# Training and test data preperation  
x_train, y_train, x_test, y_test, testfilename = prepare_data(Train_directory,Test_directory)

print(x_train.shape)
# Added Code
x_train = np.asarray(x_train) 
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols)

x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols)

x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows,img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)


print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')



 ################################
 
print ("Training set size: ", str(len(x_train)))
print ("Test set size: ", str(len(x_test)))

# --------------------------------------------------------------------

def model_generate():
    
    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows,img_cols,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
#    model.add(Dropout(0.1)) 
    #2nd convolution layer  
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
#    model.add(Dropout(0.1)) 
    #3rd convolution layer
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
#    model.add(Dropout(0.5)) 
    
    model.add(Flatten())
    
    
    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1024, activation='relu'))
    ##model.add(Dropout(0.5))
     
      
    model.add(Dense(num_classes, activation='softmax'))
    
    #------------------------------
    #batch process
    #------------------------------

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()
    return model



model = model_generate()
plot_model(model, to_file='CNN_Model.png', show_shapes=True, show_layer_names = False, rankdir = 'TB')


model_checkpoint = ModelCheckpoint("../model/CNN_model.h5", 'val_acc', verbose=1,
                                                    save_best_only=True)

model_checkpoint_loss = ModelCheckpoint("../model/CNN_model_Loss.h5", 'val_loss', verbose=1,
                                                    save_best_only=True)

log_file_path = '../log/CNN_training_log.csv'
csv_logger = CSVLogger(log_file_path, append=True)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                              patience=50, min_lr=0.0001)

callbacks = [model_checkpoint, csv_logger, model_checkpoint_loss, reduce_lr]

history = model.fit(x_train, # Features
                      y_train, # Target
                      epochs=epochs, # Number of epochs
                      verbose=1, # No output
                      batch_size=batch_size, # Number of observations per batch
                      callbacks=callbacks,
                      validation_data=(x_test, y_test)) # Data for evaluation

# --------------------------------------------------------------------
# Visualize the training and test loss through epochs

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure(figsize=(10, 8))
plt.title('Training and Validation Loss - CNN', fontsize=20)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'], fontsize=16)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.savefig('Loss_Val - CNN.png')
plt.show();


# save the loss and accuracy data
f = open('../model/CNN_history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()

model.save_weights('../model/CNN_model_Last.h5')

model.load_weights('../model/CNN_model.h5') #load weights best is Copy

y_pred = model.predict_classes(x_test)
y_true = [0] * len(y_pred)

for i in range(0, len(y_test)):
    max_index = np.argmax(y_test[i])
    y_true[i] = max_index

# --------------------------------------------------------------------
# Print wrong classifications 

for i in range(len(y_pred)):
    if(y_pred[i] != y_true[i]):
        print(str(i) + ' ' + testfilename[i] +' --> Predicted: ' +  str(y_pred[i]) + " Expected: " + str(y_true[i]))

# --------------------------------------------------------------------
# Draw the confusion matrix 
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

labels = ['Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt']

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)

# --------------------------------------------------------------------
# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

ac = float("{0:.3f}".format(scores[1]*100))
# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=labels, normalize=False,
                      title= 'Confusion Matrix - CNN , Acc = ' + str(ac))
plt.savefig('Non-Normalized CNN.png')
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=labels, normalize=True,
                      title= 'Normalized Confusion Matrix - CNN, Acc = '+ str(ac))
plt.savefig('Normalized CNN.png')
plt.show()

# --------------------------------------------------------------------
# Save the model and the weights 
model_json = model.to_json()
with open("../model/CNN_model.json", "w") as json_file:
    json_file.write(model_json)
#model.save_weights("./model/model.h5")
model.save_weights('../model/CNN_model.h5')
print("Saved model to disk")
