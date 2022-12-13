import cv2
from mtcnn.mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()
while True:
    try:
    	_, frame = cap.read()
    	faces = detector.detect_faces(frame)
    	for box in faces:
        	x, y, w, h = box['box']
        	x1, y1 = x+w, y+h
        	cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255))
    	cv2.imshow('Haar Cascade Classifier',frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
    except Exception as e:
        print(e)
        break
cap.release()
cv2.destroyAllWindows()