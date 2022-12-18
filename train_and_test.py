import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from model.classify_model import simple


data_path='/content/drive/MyDrive/knees/dataset'
categories = list(os.listdir(data_path))
labels = [i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) 
print(label_dict)
print("categories: ",categories)    #created the categories
print("Labels: ",labels)        #created the labels

##############################################################################

img_size=256

data=[]
label=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the image  into 256 x 256, since we need a fixed 
            #common size for all the images in the dataset
            data.append(resized)
            label.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)
        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here.
            # And pass to the next image


data=np.array(data)/255.0      #Normalizing the dataset between [0,1]
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
label=np.array(label)

from keras.utils import np_utils
new_label = np_utils.to_categorical(label)

#############################################################################
# data spliting 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, new_label, test_size=0.1 )

###############################################################################
# Reviewing with some input images and labels
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(x_test[i]))
    plt.xlabel(categories[np.argmax(y_test[i])])
fig1 = plt.gcf()
fig1.savefig('output/Preview.png', dpi=100)
plt.show() 

###############################################################################

num_classes = len(np.unique(labels))
print("Number of Class we have: ", num_classes)
IMG_HEIGHT = data.shape[1]
IMG_WIDTH  = data.shape[2]
IMG_CHANNELS = data.shape[3]

##############################################################################
# Calling the Model here

def get_model():
    return simple(num_classes = num_classes,
                  IMG_HEIGHT = IMG_HEIGHT,
                  IMG_WIDTH = IMG_WIDTH,
                  IMG_CHANNELS = IMG_CHANNELS)

model = get_model()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from datetime import datetime
start = datetime.now()

history = model.fit(x_train,y_train,
                    batch_size = 32,
                    epochs = 100,
                    verbose = 1,
                    validation_split = 0.2 )
stop = datetime.now() 
print("Total time to execuate is: ", stop - start)

model.save('model/Trained_simple_model.hdf5')

###############################################################################

# plot the training loss and accuracy

epochs = 100   # change this also

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")

plt.savefig("model/CNN_Model.png",dpi=300)
plt.show() 

vaL_loss, val_accuracy = model.evaluate(x_test, y_test, verbose = 1)
print("test loss:", vaL_loss,'%')
print("test accuracy:", val_accuracy,"%")

##############################################################################

# selecting the test data number is X

X = 32
img_size = 256
img_single = x_test[X]
img_single = cv2.resize(img_single, (img_size, img_size))
img_single = (np.expand_dims(img_single, 0))
img_single = img_single.reshape(img_single.shape[0],256,256,1)

predictions_single = model.predict(img_single)

plt.figure()
#plt.title('A.I predicts:',categories[np.argmax(predictions_single)])
#plt.suptitle("Correct prediction for label",np.argmax(y_test[X]),'is',categories[np.argmax(y_test[X])])

print('A.I predicts:',categories[np.argmax(predictions_single)])
print("Correct prediction for label",np.argmax(y_test[X]),'is',categories[np.argmax(y_test[X])])
plt.imshow(np.squeeze(img_single))
plt.grid(False)

fig1 = plt.gcf()
fig1.savefig('output/Test_result.png', dpi=100)
plt.show()

#############################################################################
# Confusion Matrix

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

test_labels = np.argmax(y_test, axis=1)
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=-1)

cm  = confusion_matrix(test_labels, predictions)

plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(5), ['Normal','Doubtful','Mid','Moderate','Severe'], fontsize=16)
plt.yticks(range(5), ['Normal','Doubtful','Mid','Moderate','Severe'], fontsize=16)
fig1 = plt.gcf()
fig1.savefig('output/Confusion_matrix.png', dpi=100)
plt.show()


##############################################################################
