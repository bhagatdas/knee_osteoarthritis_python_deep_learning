from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

def simple(num_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
  input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
  model=Sequential()

  model.add(Conv2D(128,(3,3),input_shape = input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64,(3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(32,(3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dropout(0.2))

  model.add(Dense(128,activation='relu'))

  model.add(Dropout(0.1))
  model.add(Dense(64,activation='relu'))

  #model.add(Dense(5,activation='softmax'))
  model.add(Dense(5,activation='sigmoid'))

  return model
