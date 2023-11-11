import cv2
import json
import numpy as np

f = open('config.json')
config = json.load(f)

cap = cv2.VideoCapture(config["TestFilePath"])

frames = []

def convert_to_greyscale(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # Find the average of the BGR pixel values
            img[i, j] = sum(img[i, j]) * 0.33
    return img


while(1):
   ret, frame = cap.read(0)
   if frame is not None:
       gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       frames.append(gray_image)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       break


frames = np.asarray(frames)
print(frames.shape)
