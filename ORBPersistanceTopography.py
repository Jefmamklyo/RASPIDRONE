import cv2 as cv
import numpy as np




#Params
cam1  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cam2 = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)


#srtyting resultion
cam1.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam1.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam1.set(cv.CAP_PROP_FPS, 15)

cam2.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam2.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam2.set(cv.CAP_PROP_FPS, 15)


checkerBoardSize = (9,6)
checkerSquareSize = 0.025 #meters

baseLine = 8 #distance between cameras cm
fx = 3015 #camera fiocal lenght foudn from tuning


print("Camera 1:",cam1.isOpened())
print("Casmera 2:", cam2.isOpened())


orb = cv.ORB_create(nfeatures=800, scoreType = cv.ORB_FAST_SCORE) #initiliase orb
#initilaise stereo 
stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 128, blockSize = 7, P1 = 1176, P2=4704, disp12MaxDiff=1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)


#Stereo Vision Helper functions

def StereoDisparity(leftCam, rightCam):
     disparity = stereo.compute(leftCam, rightCam).astype(np.float32)/16
     return disparity


def heightMapGeneration(disparity):
    validMask = (disparity > 5) & (disparity < fx * baseLine /0.15)
    heightMap = np.zeros(disparity.shape, dtype=np.uint8)

    if validMask.any():
        validValues = disparity[validMask]
        heightMap[validValues] = fx * baseLine/disparity[validValues]

        
    heightMap, validMask


#convert to RGB because why not
def RGBHeightmap(heightMap, validMask):
     coloured = cv.applyColorMap(heightMap, cv.COLORMAP_JET)
     coloured[~validMask] =[0,0,0] #mark no conidence pixeks
     return coloured

###################################################
###################################################
###################################################


#PARAMS TO BE NEEDED? PLEWAES EDIT LATER IF NOYT BE

map3d = [] #3d pointcloud map
prevFrame = None #For getting differences between frames?
step = 5
frameNumber = 0 #used with step as each frame being tracked would be incredibly inefficient so we only track every 5 ish grames
#param to store rotation adn translation

while True:


    #initial camera setup
    retL, frameLeft = cam1.read()
    retR, frameRight = cam2.read()



    grayLeft = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(frameRight, cv.COLOR_BGR2GRAY)

    

    #step counter for optimisation

    #ORB implementation
        #Fweature matching mbetween frames
        #Transformation Matrix

    #Stereo depth identification


    #Sample 3d Height maps with valid poixel coordinates from stereo maps

    #Build 3d Maps
    # 
    # #display  



    exitKey= cv.waitKey(1)

    if exitKey == ord('l'):
        break


cam1.release()
cam2.release()
            