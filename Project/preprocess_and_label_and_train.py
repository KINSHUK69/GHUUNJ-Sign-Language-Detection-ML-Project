
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# PREPROCESS DATA AND CREATE LABLES AND FEATURES

from sklearn.model_selection import train_test_split  # Help partation training data from training data and testing data
from keras.utils.np_utils import to_categorical       # convert your data into one hot talbel data


# path for exported data,numpy arrays
DATA_PATH = os.path.join('MP_Data')

# action that we try to detect
actions = np.array(['Hello', 'Thanks', 'I_Love_You', 'Like'])

'''
 actions = np.array(['Hello','Thanks','I_Love_You','Like','DisLike','Me','See_You_Later','Father','Mother','Yes','No','Help','Please','Thank_You','Want','What','Dog','Cat','Again_Or_Repeat','EatFood','Milk','More','GoTo','Bathroom','Fine','Like','Learn','Sign','Finish_Or_Done','Name','How'])
 using 30 frames to training for our action detection model  i.e., (30*30*4*1662) frames  for a single action
                                                                   (videos * frames * actions * key-points)
'''
# 30 videos worth of data
no_sequences = 30
# videos are going to be 30 frames in length
sequence_length = 30

# create label data
label_map = {label:num for num, label in enumerate(actions)}
'''
label_map     {'hello':0,'Thanks':1,'I_Love_You':2,'like':3,} 

'''


# bringing in the data
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
Y = to_categorical(labels).astype(int)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.05)

# BUILD AND TRAIN LSTM NEURAL NETWORK

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

'''

res=[.7,.2,.1,.5]               currently has 4 values as there are currently only 4 values to detect i.e., 'Hello', 'Thanks', 'I_Love_You', 'Like'  : Its called multiple class classification model 
actions[np.argmax(res)]         the result will be hello as it has the highest probability


here we are using loss = categorical_crossentropy as there are multiple class classifications

we used LSTM as it has the highest accuracy and it does not have multiple stacked neural networks 
less data required 
faster to train 
faster detection

'''
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(X_train,Y_train,epochs=2000,callbacks=[tb_callback])    #  if i do higher  echos i reach training saturation and from then the training models accuracy degrades in simpler words over training

model.summary()


# MAKING PREDICTIONS

res = model.predict(X_test)
print(actions[np.argmax(Y_test[4])])


# SAVE WEIGHTS
model.save('action.h5')

'''
IF WE DELETE OUR MODEL  we can reevaluate from recompiling the model part and  using
del model
model.load_weights('action.h5')
'''