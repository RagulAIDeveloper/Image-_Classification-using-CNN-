from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
#convolutionlayer
model.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = "relu"))
#maxpooling adding
model.add(MaxPooling2D(pool_size =(2,2)))
#fatten adding
model.add(Flatten())
#fully connected layer
model.add(Dense(units = 128,activation = "relu"))
model.add(Dense(units = 1,activation = "sigmoid"))

#compile
model.compile(optimizer ="adam",loss = "binary_crossentropy",  metrics =["accuracy"])
#image different meothd generator
train_datagen=ImageDataGenerator(rescale = 1.255,
                                 shear_range =0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True)
val_datagen =ImageDataGenerator(rescale =1.255)
#set
training_set = train_datagen.flow_from_directory("C:/Users/msrag/OneDrive/Desktop/Deep Learning/project2/images",
                                                 target_size =(64,64),
                                                 batch_size =8,
                                                 class_mode="binary")
val_set =val_datagen.flow_from_directory("C:/Users/msrag/OneDrive/Desktop/Deep Learning/project2/val",
                                              target_size =(64,64),
                                              batch_size =8,
                                              class_mode = "binary")

#training
model.fit_generator(training_set,
                    steps_per_epoch =5,
                    epochs =50,
                    validation_data =val_set,
                    validation_steps =2)
#model save
model_json = model.to_json()
with open ("model.json","w") as json_file:
 json_file.write(model_json)
model.save_weights("model.h5")
