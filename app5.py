import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create folders for cats and dogs
os.makedirs('dataset/cats', exist_ok=True)
os.makedirs('dataset/dogs', exist_ok=True)

image = cv2.imread('E:\Fiverr\Sanad112\images\Abyssinian_1.jpg')

# Check if the image is loaded successfully
if image is not None:
    # Resize the image
    new_image = cv2.resize(image, (100, 100))
    
    # Display the original and resized images
    cv2.imshow('Original Image', image)
    cv2.imshow('Resized Image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image.")
# Generate synthetic cat images
for i in range(1000):
    cat = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(f'dataset/cats/cat_{i}.jpg', cat)

# Generate synthetic dog images
for i in range(1000):
    dog = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(f'dataset/dogs/dog_{i}.jpg', dog)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocess the images and labels
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(
    'dataset',
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Train the model
model.fit(train_data_gen, epochs=10)

# Load and preprocess a new image
new_image = cv2.imread('E:\Fiverr\Sanad112\images\Abyssinian_1.jpg')
new_image = cv2.resize(new_image, (100, 100))
new_image = np.expand_dims(new_image, axis=0) / 255.0

# Classify the new image
prediction = model.predict(new_image)
if prediction[0] > 0.5:
    print("It's a Cat! ------- Machine Learning Identified Image ")
else:
    print("It's a Dog! ------- Machine Learning Identified Image ")