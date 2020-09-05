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
from keras.layers.normalization import BatchNormalization

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
from keras.utils.vis_utils import plot_model
import pickle

num_classes = 8 # 'Neutral', 'Angry','Disgust', 'Fear', 'Happy', 'Sad', Surprise', 'Contempt'
batch_size = 64
epochs = 500
fit = True
seed(8)


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
def prepare_data(TrainFileName, TestFileName):
    X, Y = [], []
    x_train, y_train, x_test, y_test = [], [], [], []
    with open(TrainFileName) as f:
        content = f.readlines()
        lines = np.array(content)
        num_of_instances = lines.size
        
        for i in range(0,num_of_instances):   
            if lines[i] != '\n':
                filename, emotion, LM, distances = lines[i].split(",")
                distances = distances.strip() 
                val = distances.split(" ")
                val = np.array(val)
                val = val.astype(np.float)
                
                ## New added code
#                val = np.expand_dims(val, axis = 0)
                #val = val.reshape(1,-1) #4624
                val = val.reshape(-1,1)
                minmax = preprocessing.MinMaxScaler()
                val = minmax.fit_transform(val)
    
                emotion = keras.utils.to_categorical(emotion, num_classes)
    
                X.append(val)
                Y.append(emotion)
                
    x_train = X
    y_train = Y
    
    X, Y, testfilename = [], [], []    
    with open(TestFileName) as f:
        content = f.readlines()
        lines = np.array(content)
        num_of_instances = lines.size
        
        for i in range(0,num_of_instances): 
            if lines[i] != '\n':
                filename, emotion, LM, distances = lines[i].split(",")
                distances = distances.strip() 
                val = distances.split(" ")
                val = np.array(val)
                val = val.astype(np.float)
                ## New added code
#                val = np.expand_dims(val, axis = 0)
#                val = val.reshape(1,-1) #4624
                val = val.reshape(-1,1)
                minmax = preprocessing.MinMaxScaler()
                val = minmax.fit_transform(val)
    
                emotion = keras.utils.to_categorical(emotion, num_classes)
    
                X.append(val)
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
x_train, y_train, x_test, y_test, testfilename = prepare_data("CKUBD_Train_DLIB_MTCNN.csv","CKUBD_Test_DLIB_MTCNN.csv")

print(x_train.shape)
# Added Code
x_train = x_train.reshape(x_train.shape[0],4624) #4624
x_test = x_test.reshape(x_test.shape[0],4624) #4624

 ################################
 
print ("Training set size: ", str(len(x_train)))
print ("Test set size: ", str(len(x_test)))
# --------------------------------------------------------------------
# Construct the NN structure
model = Sequential()
#1st layer
model.add(Dense(1024, input_shape=(4624,)))

#model.add(BatchNormalization())

model.add(Dense(1024))

#model.add(BatchNormalization())

model.add(Dense(1024))

#model.add(BatchNormalization())

model.add(Dense(1024))

#model.add(BatchNormalization())

model.add(Dense(1024))

#model.add(Dropout(0.5))

#model.add(BatchNormalization())
##additional layers
#model.add(Dense(512))
#model.add(Dense(512))
## end of additional layers

model.add(Dense(num_classes, activation='softmax'))

# compile the model
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9)
# opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
plot_model(model, to_file='ANN_Model.png', show_shapes=True, show_layer_names = False, rankdir = 'TB')


model.load_weights('../model/ANN_model1024.h5')



model_checkpoint = ModelCheckpoint("../model/ANN_model1024.h5", 'val_acc', verbose=1,
                                                    save_best_only=True)

model_checkpoint_loss = ModelCheckpoint("../model/ANN_model1024_Loss.h5", 'val_loss', verbose=1,
                                                    save_best_only=True)

log_file_path = '../log/ANN_training_log.csv'
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
plt.title('Training and Validation Loss - ANN', fontsize=20)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'], fontsize=16)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.savefig('Loss_Val - ANN.png')
plt.show();


# save the loss and accuracy data
f = open('../model/ANN_history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()

model.save_weights('../model/last_weights1024.h5')

model.load_weights('../model/ANN_model1024.h5') #load weights best is Copy

#model.load_weights('../results/ANN/model1024.h5') #load weights best is Copy

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
                      title= 'Confusion Matrix - ANN , Acc = ' + str(ac))
plt.savefig('Non-Normalized ANN.png')
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=labels, normalize=True,
                      title= 'Normalized Confusion Matrix - ANN, Acc = '+ str(ac))
plt.savefig('Normalized ANN.png')
plt.show()

# --------------------------------------------------------------------
# Save the model and the weights 
model_json = model.to_json()
with open("../model/ANN_model1024.json", "w") as json_file:
    json_file.write(model_json)
#model.save_weights("./model/model.h5")
model.save_weights('../model/ANN_model1024.h5')
print("Saved model to disk")

