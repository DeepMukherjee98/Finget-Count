from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential

import matplotlib.pyplot as plt

import numpy as np
batch_size = 128

train_datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    r'C:\Users\KIIT\Desktop\TRY IMAGES\train',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=batch_size,
    classes=['zero', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    r'C:\Users\KIIT\Desktop\TRY IMAGES\test',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=batch_size,
    classes=['zero', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

h1, _, _ = plt.hist(
    train_gen.classes, 
    bins=range(7), 
    alpha=.8, 
    color='blue', 
    edgecolor='black'
)
h2, _, _ = plt.hist(
    test_gen.classes, 
    bins=range(7), 
    alpha=.8, 
    color='red', 
    edgecolor='black'
)
plt.ylabel('Number of instances')
plt.xlabel('Class')
X, y = train_gen.next()
print(X.shape, y.shape)

plt.figure(figsize=(16,16))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.title('Label: %d' % np.argmax(y[i]))
    img = np.uint8(255*X[i, :, :, 0])
    plt.imshow(img, cmap='gray')


model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5')
]








history = model.fit_generator(
    train_gen,
    steps_per_epoch=120,
    epochs=40,
    validation_data=test_gen,
    validation_steps=28,
    callbacks=callbacks
)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], 'r-', label='train')
plt.plot(range(epochs), history.history['val_loss'], 'b-', label='test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.subplot(1,2,2)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['acc'], 'r-', label='train')
plt.plot(range(epochs), history.history['val_acc'], 'b-', label='test')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

 
model.save("model_1.h5")