

# https://www.kaggle.com/gpreda/classifying-cursive-hiragana-mnist-using-cnn
import warnings
warnings.filterwarnings('ignore')


# In[74]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


# seaborn color palette 
color = sns.color_palette()

# For REPRODUCIBILITY
seed = 111
np.random.seed(seed)
tf.set_random_seed(seed)


# In[75]:



kmnist_train_images_path = "kuzushiji/kmnist-train-imgs.npz"
kmnist_train_labels_path = "kuzushiji/kmnist-train-labels.npz"

# Path to the test images and corresponding labels
kmnist_test_images_path = "kuzushiji/kmnist-test-imgs.npz"
kmnist_test_labels_path = "kuzushiji/kmnist-test-labels.npz"

# Load the training data from the corresponding npz files
kmnist_train_images = np.load(kmnist_train_images_path)['arr_0']
kmnist_train_labels = np.load(kmnist_train_labels_path)['arr_0']

# Load the test data from the corresponding npz files
kmnist_test_images = np.load(kmnist_test_images_path)['arr_0']
kmnist_test_labels = np.load(kmnist_test_labels_path)['arr_0']

train_images = np.load('kuzushiji/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('kuzushiji/kmnist-test-imgs.npz')['arr_0']
train_labels = np.load('kuzushiji/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('kuzushiji/kmnist-test-labels.npz')['arr_0']


# In[52]:


# Get the unique labels
labels = np.unique(kmnist_train_labels)

# Get the frequency count for each label
frequency_count = np.bincount(kmnist_train_labels)

# Visualize 
plt.figure(figsize=(10,5))
sns.barplot(x=labels, y=frequency_count);
plt.title("Distribution of labels in KMNIST training data", fontsize=16)
plt.xlabel("Labels", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[53]:


char_df = pd.read_csv('kuzushiji/kmnist_classmap.csv', encoding = 'utf-8')


# In[54]:


def plot_sample_images_data(images, labels):
    plt.figure(figsize=(12,12))
    for i in range(10):
        imgs = images[np.where(labels == i)]
        lbls = labels[np.where(labels == i)]
        for j in range(10):
            plt.subplot(10,10,i*10+j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imgs[j], cmap=plt.cm.binary)
            plt.xlabel(lbls[j])


# In[55]:


plot_sample_images_data(train_images, train_labels)


# In[56]:


# data preprocessing
def data_preprocessing(images, labels):
    out_y = keras.utils.to_categorical(labels, 10)
    num_images = images.shape[0]
    x_shaped_array = images.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


# In[57]:


X, y = data_preprocessing(train_images, train_labels)
X_test, y_test = data_preprocessing(test_images, test_labels)


# In[58]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2019)


# In[59]:


#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[60]:


#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)


# In[62]:


#predict first 4 images in the test set
model.predict(X_test[:4])


# In[63]:


#actual results for first 4 images in test set
y_test[:4]


# In[64]:


#get the predictions for the test data
predicted_classes = model.predict_classes(X_val)
#get the indices to be plotted
y_true = np.argmax(y_val,axis=1)


# In[65]:


correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]


# In[66]:


print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])


# In[67]:


target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item().encode('ascii', 'ignore')) for i in range(10)]

print(classification_report(y_true, predicted_classes, target_names=target_names))


# In[70]:


def plot_images(data_index,cmap="Blues"):
    # Plot the sample images now
    f, ax = plt.subplots(5,5, figsize=(12,12))

    for i, indx in enumerate(data_index[:25]):
        ax[i//5, i%5].imshow(X_val[indx].reshape(28,28), cmap=cmap)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("True:{}  Pred:{}".format(y_true[indx],predicted_classes[indx]))
    plt.show()    

plot_images(incorrect, "Reds")


# In[71]:


#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)
#get the indices to be plotted
y_true = np.argmax(y_test,axis=1)
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])
target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item().encode('ascii', 'ignore')) for i in range(10)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

