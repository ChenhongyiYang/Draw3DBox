import cv2 as cv
import numpy as np





def read_video(filename='/Users/yangchenhongyi/Documents/3d_box/own_3d/1543865088765998.mp4'):
    cap = cv.VideoCapture(filename)
    count = 0
    while cap.isOpened() and count < 420:
        ret, frame = cap.read()
        #img = cv.resize(frame,(1224,370))
        img = frame
        cv.imwrite('videos/%d.png'%count,img)
        count += 1
        #cv.imshow('video',frame)
        #cv.waitKey(150)











if __name__ == '__main__':
    read_video()














































