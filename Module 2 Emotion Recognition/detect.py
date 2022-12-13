import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
#%matplotlib inline

dataset = "../Datasets/"

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral","Sad","Surprised"]

model_json_file = "model.json"
model_weights_file = "model_weights.h5"

with open(model_json_file,"r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(model_weights_file)

def predict_emotion(img):
        preds = loaded_model.predict(img)
        return EMOTIONS_LIST[np.argmax(preds)]

classifier = cv2.CascadeClassifier(dataset+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(frame)
    for box in faces:
        x,y,w,h = box
        x1,y1 = x+w,y+h
        fc = gray[y:y1,x:x1]
        roi = cv2.resize(fc, (48, 48))
        result = predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(frame, result, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_8)	
        cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
    cv2.putText(frame, "Press \'q\' to quit", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2,cv2.LINE_8)	
    cv2.imshow("Emotional Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q' ):
        break

cap.release()
cv2.destroyAllWindows()
