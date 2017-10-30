#skin Segmentation logic from https://github.com/mrgloom/skin-detection-example

#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#face.png image from http://graphics.cs.msu.ru/ru/node/899

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import os
from math import atan2,degrees
import csv
from sklearn import tree
from sklearn.cross_validation import train_test_split

global_cols = 0
final_result = []
def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)
    return clf

def GetAngleOfLineBetweenTwoPoints(p1, p2):
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        return degrees(atan2(yDiff, xDiff))

def cntrAngle(cnt):
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((global_cols-x)*vy/vx)+y)
    
    return GetAngleOfLineBetweenTwoPoints([0,lefty], [global_cols-1,righty])

def ApplyToImage(path, flUseHSVColorspace):
    
    data, labels= ReadData()
    clf= TrainTree(data, labels, flUseHSVColorspace)
    
    im = Image.open('/Users/hshobak/Work/uw/css581/project/imgs-ML-Driver-Distraction/test/c0/' + path)
    #contrast = ImageEnhance.Contrast(im)
    #im = contrast.enhance(2)
    #Brightness = ImageEnhance.Brightness(im)
    #im = Brightness.enhance(1)
    Sharpness = ImageEnhance.Sharpness(im)
    im = Sharpness.enhance(2)
    #Color = ImageEnhance.Color(im)
    #im = Color.enhance(2)
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    
    # detect eye(s)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray)
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))

    #cv2.imshow('im',imgLabels)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if (flUseHSVColorspace):
        cv2.imwrite('result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
        ret,thresh = cv2.threshold(cv2.GaussianBlur(cv2.imread("result_HSV.png"), (25, 25), 0),50,255,0)
        # place eyes too
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #cv2.imwrite('/Users/hshobak/Work/uw/css581/project/imgs-ML-Driver-Distraction/train/c0-eye-arm' + path, thresh)
        #cv2.imshow('im', thresh)
    else:
        cv2.imwrite('result_RGB-test.png',((-(imgLabels-1)+1)*255))
        ret,thresh = cv2.threshold(cv2.GaussianBlur(cv2.imread('result_RGB-test.png'), (25, 25), 0),50,255,0)
        #print thresh.dtype
        _,contours,hierarchy = cv2.findContours(cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY), 1, 2)
        rows,cols = thresh.shape[:2]
        global_cols = cols
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = sorted(contours, key=cntrAngle, reverse=True)
        del contours[1:]
        
        potential_results = [(0,0)]
        for cnt in contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #eye_arm_angle = int(math.atan((cY-eyes[0][1])/(eyes[0][0]-cX))*180/math.pi)
            if not len(eyes):
                continue
            eye_arm_angle = GetAngleOfLineBetweenTwoPoints([cX,cY],[eyes[0][0],eyes[0][1]])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            box = np.int0(box)
            im = cv2.drawContours(thresh,[box],0,(0,0,255),2)
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            
            img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
            img = cv2.line(img,(cX, cY),(eyes[0][0],eyes[0][1]),(0,255,0),2)
            
            arm_angle = GetAngleOfLineBetweenTwoPoints([0,lefty], [cols-1,righty])

            final_result.append((path, arm_angle, eye_arm_angle, "1"))

    print path

#---------------------------------------------
files = os.listdir('/Users/hshobak/Work/uw/css581/project/imgs-ML-Driver-Distraction/test/c0')
for f in files:
    try:
        ApplyToImage(f, False)
    except:
        print "failed to process: " + f 
        pass
with open('c0-test.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['path','arm_angle', 'eye_arm_angle', 'class'])
    for row in final_result:
        csv_out.writerow(row)