from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trainimgset = ImageDataGenerator(rescale=None,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
valimgset = ImageDataGenerator(rescale=1. / 255)
traindtatset = trainimgset.flow_from_directory('dataset1/Training',
                                               target_size=(128, 128),
                                               batch_size=32,
                                               class_mode='categorical')
lable1 = traindtatset.class_indices
# print(lable1)
valdtatset = valimgset.flow_from_directory('dataset1/Testing',
                                           target_size=(128, 128),
                                           batch_size=32,
                                           class_mode='categorical'
                                           )
lable2 = valdtatset.class_indices
s = len(traindtatset)
e = len(valdtatset)
print(lable2)
print(s, e)
model.fit(traindtatset, steps_per_epoch=s, epochs=e, validation_data=valdtatset, validation_steps=300)

json = model.to_json()

with open("model2.json", 'w') as json_files:
    json_files.write(json)
model.save_weights('model2.h5')
print("train successfully")
