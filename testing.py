import cv2 as cv
import numpy as np
import random
from sklearn.linear_model import RANSACRegressor, LinearRegression


cam1  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cam2 = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)


#srtyting resultionfoca 
cam1.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam1.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam1.set(cv.CAP_PROP_FPS, 15)
cam1.set(cv.CAP_PROP_AUTOFOCUS, 0)

cam2.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cam2.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cam2.set(cv.CAP_PROP_FPS, 15)
cam2.set(cv.CAP_PROP_AUTOFOCUS, 0)

stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 128, blockSize = 7, P1 = 1176, P2=4704, disp12MaxDiff=1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)


"""Plan
Capture a set amount of frames in a function and define a variable "depth" valyue param
    Account for exposure
    use python input to wait until we are at a certain distance as we are passing the distances in through a list
    Only garabnb frames passing a acceptance ciretria
    append to results
    REMOVE OUTLIERS!!!

Then tune the fx based on those images
    order by distance [proably in an array whic you p[ass into captureFrames function]
    appnd disparity and depth into a object
    convert into np arraus for cacluations
    Some alogrithm to convert find the best focal lenghts
        Maybe sqaure regrssion, standard distributions to get means and a range within 1sd, 

Maybe test on non-testing envs?
"""


distanceInputs = [0.5, 1, 1.5] #m, tune fx

results = [] #(3 object tuple [dist,disp,fx])


fx = 3050
baseline = 0.065

def MAD(x, mSD = 2):
    median = np.median(x)

    mad = 1.4826 * np.median(np.abs(x-median))


    if mad == 0:
        return x
    valid = np.abs(x-median) <= (mSD* mad)
    return x[valid]

def capture(depth):

    input(f"Place at {depth} and then run")
    
    #disacrd for expsure
    for i in range (15):
        cam1.read()
        cam2.read()
    #calc disparitr
    disparityRes = []

    #Only capture valid frames
    while len(disparityRes) > 100:
        ret1, leftFrame = cam1.read()
        ret2, rightFrame = cam2.read()
        grayLeft = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(rightFrame, cv.COLOR_BGR2GRAY)

        section = stereo.compute(grayLeft, grayRight).astype(np.float32)/16
        sSection = section[95:145, 135:185] #grab 40x40 image center 

        #applyt valid mask
        valid = sSection[sSection > 1]

        if valid < 100: #arbitratyu
             disparityRes.append(np.median(valid))
    return MAD(np.array(disparityRes))
################################
        #TUNE fx
################################




depths = np.array(distanceInputs)
np.random.shuffle(depths)


disps = np.array([capture(depth).mean() for depth in depths ])


#ransac 
u = (baseline/depths).reshape(-1,1)
ransac = RANSACRegressor(LinearRegression(fit_intercept=False), min_samples = 1, residual_threshold=  0.1, random_state=0)

ransac.fit(u,disps)

fx  = ransac.estimator_.coef_[0]

print(f"fx = {fx:.1f} ")
