import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Activation
from tensorflow.python.keras.optimizers import Adam



learning_rate = 0.001
training_epoch = 20
batch_size = 128
display_step = 10

n_input = 28  #輸入像素點
n_step = 28   #時間序列
n_hidden = 128
n_classes = 10

(x_Train,y_Train),(x_Test,y_Test)=keras.datasets.mnist.load_data()
x_Train = x_Train.reshape(-1,n_step,n_input)
x_Test = x_Test.reshape(-1,n_step,n_input)
x_Train = x_Train.astype('float32')
x_Test = x_Test.astype('float32')
x_Train/=255
x_Test/=255
y_Train_onehot=np_utils.to_categorical(y_Train)
y_Test_onehot=np_utils.to_categorical(y_Test)

model=Sequential()
model.add(LSTM(n_hidden,batch_input_shape=(None,n_step,n_input),unroll=True))
model.add(Dense(n_classes,activation='softmax'))
adam = Adam(lr=learning_rate)
model.summary()
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_Train,y_Train_onehot,batch_size=batch_size,epochs=training_epoch,verbose=2)
scores = model.evaluate(x_Test,y_Test_onehot,verbose=2)
print('LSTM_MINI test scores = ',scores[0])
print('LSTM_MINI test acc = ',scores[1])