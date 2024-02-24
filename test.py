from keras.models import  model_from_json
import numpy as np
from keras import utils

#loaded the model

json_file =open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model =model_from_json(loaded_model_json)
model.load_weights('model.h5')
print('Loaded model from disk')

def classify(img_file):
    #loaded image
    test_image =utils.load_img(img_file,target_size=(64,64))
    #convert the array
    test_image = utils.img_to_array(test_image)
    #expand the dimsions
    test_image = np.expand_dims(test_image, axis=0)
    #predict the output
    result =model.predict(test_image)

    if result[0][0] == 0:
        prediction = 'Apple'
    else:
        prediction ='Orange'
    print(prediction,img_file)

import os
path ='C:/Users/msrag/OneDrive/Desktop/Deep Learning/project2/test'
files=[]
for r,d,f in os.walk(path): # r=root,d=directories ,f=files
    for file in f:
        if '.jpeg' in file:
            files.append(os.path.join(r,file))
            
for f in files:
    classify(f)
    
    
