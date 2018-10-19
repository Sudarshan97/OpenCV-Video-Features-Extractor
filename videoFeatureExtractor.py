
import numpy as np
import cv2
from matplotlib import pyplot as plt

def detector(img):


    # img = cv2.resize(img,(1000,700))
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp , des = orb.compute(img,kp)

    img2 = cv2.drawKeypoints(img , kp ,outImage = np.array([]) ,color = (0,255,0),flags = 0)
    return img2



#trying to capture a video
def captureVideo():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = detector(gray)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


captureVideo()
