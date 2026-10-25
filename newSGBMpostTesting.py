import cv2 as cv
import numpy as np

####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO
####OLD SGBM FOR VIDEO



cam1  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cam2 = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)


#srtyting resultionfoca 
cam1.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam1.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam1.set(cv.CAP_PROP_FPS, 15)

cam2.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam2.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam2.set(cv.CAP_PROP_FPS, 15)

baseLine = 0.065 #distance between cameras cm
fx = 3015 #camera fiocal lenght foudn from tuning


print("Camera 1:",cam1.isOpened())
print("Casmera 2:", cam2.isOpened())

def StereoDisparity(leftCam, rightCam):
     stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 128, blockSize = 7, P1 = 1176, P2=4704, disp12MaxDiff=1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)
     disparity = stereo.compute(leftCam, rightCam).astype(np.float32)/16
     return disparity


def heightMapGeneration(disparity):
    validMask = disparity > 5
    heightMap = np.zeros(disparity.shape, dtype=np.uint8)

    if validMask.any():
        validValues = disparity[validMask]

        #edgecases
        dispMin, dispMax = validValues.min(), validValues.max()

        if dispMax > dispMin:
            normalised = (validValues - dispMin)/(dispMax - dispMin) * 255
            heightMap[validMask] = normalised.astype(np.uint8)
        else: #all are valid pixes with same deopt
            heightMap[validMask] = 255
    return heightMap, validMask


#convert to RGB because why not
def RGBHeightmap(heightMap, validMask):
     coloured = cv.applyColorMap(heightMap, cv.COLORMAP_JET)
     coloured[~validMask] =[0,0,0] #mark no conidence pixeks
     return coloured




stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 128, blockSize = 7, P1 = 1176, P2=4704, disp12MaxDiff=1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)



while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        

        if not ret1 or not ret2:
            print("No return value. breaking Camrea")
            break
        



        grayCam1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        grayCam2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        section = stereo.compute(grayCam1, grayCam2).astype(np.float32)/16
        sSection = section[95:145, 135:185] #grab 40x40 image center 

        cv.imshow("ssec", sSection)
        cv.imshow("aSDF", section)
        x = grayCam1[95:145, 135:185]
        cv.imshow("asdf", x)

        exitKey= cv.waitKey(1)
        idle = "F"
        if exitKey == ord('l'):
            break


cam1.release()
cam2.release()
            