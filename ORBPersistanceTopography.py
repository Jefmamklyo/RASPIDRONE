import cv2 as cv
import numpy as np




#Params
cam1  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cam2 = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)

camHeight = 240
camWidth = 320

#srtyting resultion
cam1.set(cv.CAP_PROP_FRAME_WIDTH, camWidth)
cam1.set(cv.CAP_PROP_FRAME_HEIGHT, camHeight)
cam1.set(cv.CAP_PROP_FPS, 15)
#
cam2.set(cv.CAP_PROP_FRAME_WIDTH, camWidth)
cam2.set(cv.CAP_PROP_FRAME_HEIGHT, camHeight)
cam2.set(cv.CAP_PROP_FPS, 15)


#checkerBoardSize = (9,6)
#checkerSquareSize = 0.025 #meters

baseLine = 8 #distance between cameras cm
fx = 3015 #camera fiocal lenght foudn from tuning


print("Camera 1:",cam1.isOpened())
print("Casmera 2:", cam2.isOpened())


#initilaise stereo 
stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 128, blockSize = 7, P1 = 1176, P2=4704, disp12MaxDiff=1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)


#Stereo Vision Helper functions

def StereoDisparity(leftCam, rightCam):
     disparity = stereo.compute(leftCam, rightCam).astype(np.float32)/16
     return disparity


def heightMapGeneration(disparity):
    #remove dispaority >1
    validMask = (disparity > 1 ) & (disparity < fx * baseLine /0.15)
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
position = np.eye(4, dtype = np.float32)

orb = cv.ORB_create(nfeatures=800, scoreType = cv.ORB_FAST_SCORE) #initiliase orb
bf = cv.BGFMatcher(cv.NOORM_HAMMING, crossCheck = True)

while True:


    #initial camera setup
    retL, frameLeft = cam1.read()
    retR, frameRight = cam2.read()

    #step counter for optimisation
    frameNumber += 1

    if frameNumber % step == 1:
        continue


    grayLeft = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(frameRight, cv.COLOR_BGR2GRAY)

    ########################################
    #ORB implementation
        #Fweature matching mbetween frames
        #Transformation Matrix
    ########################################
    
    kp, description = orb.detectAndCompute(grayLeft, None)

    #gather previosu frame for comparisoin
    if prevFrame is not None and description is not None:
        kpPrev, descriptionPrev = orb.detectAndComput(prevFrame, None)
        #feature matching algorith
        if descriptionPrev is not None:
            #faeture matching algorithm

            matches = bf.match(description, descriptionPrev) #match descriptions between frames
            matches = sorted(matches, key= lambda x:x.distance)

            if len(matches) > 10:
                points1 = np.float32([kp[m.queryIDX].pt for m in matches[:100]])
                points2 = np.float32([kpPrev[m.queryIDX].pt for m in matches[:100]])

                #describes current to prevoius points rotational and translational directions
                #camw/2,camheiht/ = image centre
                #last two argyuments mean confidense and RANSAC Subset aggremenet to one pixel
                essentialMatrix, mask = cv.findEssentialMat(points1, points2, fx, (camWidth/2, camHeight/2), 0.999, 1)


                #essential mmatrix multiplacation look at docuimentation
                if essentialMatrix is not None:
                    retVal, Rotation, Translation, _ = cv.recoverPose(essentialMatrix, points1, points2, focal = fx, pp = (camWidth/2,camHeight/2))

                    transformationMatrix = np.eye(4, dtype=np.float32)
                    transformationMatrix[:3, :3] = Rotation.astype(np.float32)
                    transformationMatrix[:3, 3] = Translation[:,0].astype(np.float32)
                    position = transformationMatrix @ position #imporatbnt order T@P not P@T 

    prevFrame = grayLeft.copy()

    
        
          




    #Stereo depth identification


    disparity = StereoDisparity(grayLeft, grayRight)
    depth, valid = heightMapGeneration(disparity)

    #keep values between 0.1 and 5m
    valid &= (depth > 0.1) and (depth <5)


    #Sample 3d Height maps with valid poixel coordinates from stereo maps

    #Build 3d Maps
    


    # #display  




    exitKey= cv.waitKey(1)

    if exitKey == ord('l'):
        break


cam1.release()
cam2.release()
            