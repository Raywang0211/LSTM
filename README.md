# LSTM

Through mnist to practice how to construct the LSTM model.

Here used the Keras and tensorflowGPU to achieve the goal.
The mnist is a hand write dataset that mainly used number.


import keras
from keras.utils import np_utils =====>> For one hot encoding
from keras.models import Sequential =====>> For using sequential model
from keras.layers import Dense,Dropout,LSTM,Activation ======>> the layers that i used
from tensorflow.python.keras.optimizers import Adam ====>> the optimizers that i used

n_input = 28  #that is the input data (y axis number )
n_step = 28   #time stpe which LSTM need to initial setting
n_hidden = 128 
n_classes = 10 # target class number

(x_Train,y_Train),(x_Test,y_Test)=keras.datasets.mnist.load_data()  #load the dataset 
x_Train = x_Train.reshape(-1,n_step,n_input)  #reshape the data to (non,time_step,input)
x_Test = x_Test.reshape(-1,n_step,n_input)  
x_Train = x_Train.astype('float32')       %assign the data type of the data
x_Test = x_Test.astype('float32')
x_Train/=255  %normalize
x_Test/=255
y_Train_onehot=np_utils.to_categorical(y_Train) %one hot encogin
y_Test_onehot=np_utils.to_categorical(y_Test)

model=Sequential()     #initial sequential model
model.add(LSTM(n_hidden,batch_input_shape=(None,n_step,n_input),unroll=True)) # add LSTM layer
model.add(Dense(n_classes,activation='softmax'))  #output layer add a fully connected layer with softmax activation function to classify
adam = Adam(lr=learning_rate) # setting the learning rate of the optimizer
model.summary() # show the structure of the sequential model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])  #setting the training parameter

model.fit(x_Train,y_Train_onehot,batch_size=batch_size,epochs=training_epoch,verbose=2) #training the model
scores = model.evaluate(x_Test,y_Test_onehot,verbose=2) #evaluate the model

