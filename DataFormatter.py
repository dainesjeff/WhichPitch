import cv2
import json


f = open('config.json')
config = json.load(f)

cap = cv2.VideoCapture("v2.mp4")
ret, frames = cap.read()


frames = []

while(1):
   ret, frame = cap.read()
   frames.append(frame)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       break
