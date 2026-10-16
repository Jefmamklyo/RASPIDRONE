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

baseLine = 0.08 #distance between cameras m
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

    validMask = (disparity > 1 ) 

    depth = np.zeros(disparity.shape, dtype=np.float32)

    depth[validMask] = fx * baseLine/disparity[validMask]

    validMask &= (depth >1) & (depth <5)

        
    return depth, validMask


#convert to RGB because why not
def RGBHeightmap(heightMap, validMask):
     coloured = cv.applyColorMap(heightMap, cv.COLORMAP_JET)
     coloured[~validMask] =[0,0,0] #mark no conidence pixeks
     return coloured

###################################################
###################################################
###################################################


#PARAMS TO BE NEEDED? PLEWAES EDIT LATER IF NOYT BE

map3D = [] #3d pointcloud map
prevFrame = None #For getting differences between frames?
step = 5
frameNumber = 0 #used with step as each frame being tracked would be incredibly inefficient so we only track every 5 ish grames
position = np.eye(4, dtype = np.float32)

orb = cv.ORB_create(nfeatures=800, scoreType = cv.ORB_FAST_SCORE) #initiliase orb
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

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
        kpPrev, descriptionPrev = orb.detectAndCompute(prevFrame, None)
        #feature matching algorith
        if descriptionPrev is not None:
            #faeture matching algorithm

            matches = bf.match(description, descriptionPrev) #match descriptions between frames
            matches = sorted(matches, key= lambda x:x.distance)

            if len(matches) > 10:
                points1 = np.float32([kp[m.queryIdx].pt for m in matches[:100]])
                points2 = np.float32([kpPrev[m.queryIdx].pt for m in matches[:100]])

                #describes current to prevoius points rotational and translational directions
                #camw/2,camheiht/ = image centre
                #last two argyuments mean confidense and RANSAC Subset aggremenet to one pixel
                essentialMatrix, mask = cv.findEssentialMat(points1, points2, fx, (camWidth/2, camHeight/2), cv.RANSAC, 0.999, 1)


                #essential mmatrix multiplacation look at docuimentation
                if essentialMatrix is not None:
                    retVal, Rotation, Translation, _ = cv.recoverPose(essentialMatrix, points1, points2, focal = fx, pp = (camWidth/2,camHeight/2))

                    transformationMatrix = np.eye(4, dtype=np.float32)
                    transformationMatrix[:3, :3] = Rotation.astype(np.float32)
                    transformationMatrix[:3, 3] = Translation[:,0].astype(np.float32)
                    position = position @ transformationMatrix #imporatbnt order P@T not T@P 

    prevFrame = grayLeft.copy()

    
        
          



    ########################################
    #Stereo depth identification
    ########################################

    disparity = StereoDisparity(grayLeft, grayRight)
    depth, valid = heightMapGeneration(disparity)

    #keep values between 0.1 and 5m
    valid &= (depth > 0.1) & (depth <5)

    ###############################################################################
    #Sample 3d Height maps with valid poixel coordinates from stereo maps
    ################################################################################

    ys, xs = np.where(valid)

    xs = xs[::8] #optimsation store every 8 only
    ys= ys[::8]

    z = depth[ys,xs]
    x = (xs - camWidth /2) * z / fx
    y = (ys - camHeight /2) *z / fx

    points = np.column_stack((x,y,z))


    #rigid body transformations
    pointsHomogenous = np.column_stack((points, np.ones(len(points)))) #turn pooints into 4x4 homogenous for position @ 
    worldRef = (position @ pointsHomogenous.T).T[:, :3]  #T = transpose matrix

    map3D.extend(worldRef.tolist())




    ################################################################################
    #Build 3d Maps
    ################################################################################

    #accesing a object that might not be populated/initilised (null checking/DID)
    if map3D:
        #convert to np.array
        pts = np.asarray(map3D) 


        # X is X, Z is Y, Y is Z becuase top down pershpective shift
        pointsX = ((pts[:,0]+2) * 100).astype(int)       #OG X
        pointsY = ((pts[:,2]+2)*100).astype(int) #OG Z
        height = pts[:,1] #OG Y

        heightmap = np.zeros((400,400), np.float32) #initilise the 400 by 400 height map


        #& = bitswise and, px and py < 400 not <= due to 0 based indexing
        filter = (pointsX >= 0 ) & (pointsX < 400) & (pointsY >= 0) & (pointsY < 400)

        #apply filter
        pointsX, pointsY, heights = pointsX[filter], pointsY[filter], height[filter]

        #sort by heighest + populate

        sortedIDX = np.argsort(heights)
        heightmap[pointsY[sortedIDX], pointsX[sortedIDX]] = heights[sortedIDX]

        #normalise between 0-255 fo4r visualsation + colormap

        view = cv.normalize(heightmap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        view = cv.applyColorMap(view, cv.COLORMAP_JET)
        
    else:
        #create blacnk fallback image
        view = np.zeros((400,400,3), np.uint8)

    ################################################################################      
    # #display  
    ################################################################################


    cv.putText(view, f"3DPoints = {len(map3D)}", (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv.imshow("Live Topography", view)
    cv.imshow("Left Camera", grayLeft)


    exitKey= cv.waitKey(1)

    if exitKey == ord('l'):
        break

################################################################################      
# #Save  for persistance 
################################################################################



cam1.release()
cam2.release()
cv.destroyAllWindows()            

np.save("points.npy", np.asarray(map3D))