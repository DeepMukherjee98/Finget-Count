import cv2
import numpy as np



cap_region_x_begin=0.5 
cap_region_y_end=0.8 
threshold = 25  
blurValue = 41  
bgSubThreshold = 50
learningRate = 0

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

capture = cv2.VideoCapture(0)
i=0
while(True):
     
    ret, frame = capture.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('Original', frame)
    img = removeBG(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
                int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
     # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', thresh)
    cv2.imwrite(r'C:\Users\KIIT\Desktop\TRY IMAGES\test\setttttttttt\set'+str(i)+'.jpg',thresh)
    i+=1
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()
  