import cv2

from classification.test import classfy
from segmentation.test import tumor_segmentation
from flask import Flask,request,jsonify
import base64
import numpy as np
from Chatbot import chatbot



# from segmentation.metrics import dice_coef, dice_loss

app=Flask(__name__)

@app.route('/',methods=['GET'])
def welcome():
    return "Welcome TO Our Brain Tumor Detection Project"

@app.route('/api/briantumor',methods=['POST'])
def tumor_detection():
    data = request.get_json();
    img = data['img']
    imgbytes = base64.b64decode(img)
    imgbytes= np.frombuffer(imgbytes, np.uint8)
    imgbytes =cv2.imdecode(imgbytes,cv2.IMREAD_COLOR)
    classfication,confidence_level=classfy(imgbytes)
    print(classfication)
    print(confidence_level)
    if True:
        sengmentaiton=tumor_segmentation(imgbytes)
        _, sengmentaiton = cv2.imencode('.jpg', sengmentaiton)
        sengmentaiton=cv2.imdecode(sengmentaiton, cv2.IMREAD_COLOR)
        sengmentaiton=cv2.cvtColor(sengmentaiton,cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(sengmentaiton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tumor_size = cv2.countNonZero(sengmentaiton)
        in_cm = tumor_size * 0.01;
        sengmentaiton=cv2.cvtColor(sengmentaiton,cv2.COLOR_GRAY2RGB)
        for contour in contours:
            tumor_area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if(tumor_area>5):
                cv2.rectangle(sengmentaiton, (x, y), (x + w, y + h), (255, 23, 0), 2)
        _, sengmentaiton = cv2.imencode('.jpg', sengmentaiton)
        sengmentaiton = base64.b64encode(sengmentaiton).decode('utf-8')
        return jsonify({'classify':classfication ,
                        'segmantation':sengmentaiton,
                        'size':in_cm,
                        'confidence_level':confidence_level})
    # else:
    #     return jsonify({'classify':classfication ,
    #                     'segmantation':"",
    #                     'size':0,
    #                     'confidence_level':confidence_level})





@app.route("/api/chat",methods=['POST'])
def chatbotx():
    data = request.get_json();
    text = data['send']
    return jsonify(
        {'respons':chatbot.process(text)}
    )



if __name__=='__main__':
    app.run(debug=True,port=8000)
