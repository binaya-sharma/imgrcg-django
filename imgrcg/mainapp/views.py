from django.shortcuts import render
# Create your views here.

import numpy as np
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph

img_height, img_width=100, 100
with open('./imgrcg_model/model.json','r') as f:
    labelInfo=f.read()

labelInfo = json.loads(labelInfo)
target_size=(100, 100)
# batch_size=batch_size

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('./imgrcg_model/model.h5')

# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

# Create your views here.

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileobj =request.FILES['filePath']
    # print (request.FILES['filePath'])
    fs=FileSystemStorage()
    filePathName=fs.save(fileobj.name,fileobj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName

    
    test_image = image.load_img(testimage, target_size = (100,100))
    # imageplot = plt.imshow(test_image)
    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    #test_image = image.img_to_array(test_image)
    # result = model.predict(x)
    with model_graph.as_default():
        with tf_session.as_default():
            result=model.predict(images)
            

    predictionL=result

    context={'filePathName':filePathName,'predictionL':predictionL}
    return render(request,'index.html',context)