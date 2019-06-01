# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 15, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 300)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/40.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = 'Bags'
elif result[0][1] >= 0.5:
    prediction = 'bike'
elif result[0][2] >= 0.5:
    prediction = 'buses'
elif result[0][3] >= 0.5:
	prediction = 'car'
elif result[0][4] >= 0.5:
    prediction = 'cats'
elif result[0][5] >= 0.5:
    prediction = 'chair'
elif result[0][6] >= 0.5:
	prediction = 'cycle'
elif result[0][7] >= 0.5:
    prediction = 'dogs'
elif result[0][8] >= 0.5:
    prediction = 'dresses'
elif result[0][9] >= 0.5:
	prediction = 'flowers'
elif result[0][10] >= 0.5:
    prediction = 'laptops'
elif result[0][11] >= 0.5:
    prediction = 'mobiles'
elif result[0][12] >= 0.5:	
	prediction = 'specticals'
elif result[0][13] >= 0.5:
	prediction = 'tcups'
elif result[0][14] >= 0.5:	
	prediction = 'truck'
else:
    prediction = 'Cannot predict'
print(result.shape)
print(result)
print(prediction)
classifier.save('model.h5')
