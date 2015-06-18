import numpy as np
import cv2

print 'Program Start...'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH)

number = ["01","02","03","04","05","06","07","08","09","10"
,"11","12","13","14","15","16","17","18","19","20"
,"21","22","23","24","25","26","27","28","29","30" ]

nBoardW=8
nBoardH=6

objp = np.zeros((nBoardW*nBoardH,3), np.float32)
objp[:,:2] = np.mgrid[0:nBoardW,0:nBoardH].T.reshape(-1,2)

R = np.asarray([(0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
T = np.asarray([(0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
E = np.asarray([(0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
F = np.asarray([(0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])

objpoints = [] # 3d point in real world space
Right_imgpoints = [] # 2d points in image plane.
Left_imgpoints = [] # 2d points in image plane.

samples_to_use = 30

height = 0
width = 0

for i in range(0, samples_to_use): 
    
    print 'Reading frame %s of %s' % (i+1, samples_to_use)
    
    strImg_R ="Right"
    strImg_L = "Left"
    strImg_N = number[i]
    strFormat = ".jpg"
    
    strImg_R = strImg_R + strImg_N + strFormat
    strImg_L = strImg_L + strImg_N + strFormat
        
    Right_im = cv2.imread(strImg_R)
    Left_im = cv2.imread(strImg_L)
    
    height, width, depth  = Right_im.shape
    
    print 'finding chessboard'
    
    R_ret , R_corners = cv2.findChessboardCorners(Right_im, (nBoardW,nBoardH), None)
    L_ret , L_corners = cv2.findChessboardCorners(Left_im, (nBoardW,nBoardH), None)
     
    if R_ret == True and L_ret == True:
        print 'Right_found chessboard'
        objpoints.append(objp)  
        Right_imgpoints.append(R_corners)
        Left_imgpoints.append(L_corners)
        
        print 'samples: %i' % len(Right_imgpoints)
        print 'samples: %i' % len(Left_imgpoints)

        cv2.drawChessboardCorners(Right_im, (nBoardW,nBoardH), R_corners, R_ret)
        cv2.drawChessboardCorners(Left_im, (nBoardW,nBoardH), L_corners, L_ret)        
        
        cv2.imshow("Right_im",Right_im)
        cv2.imshow("Left_im",Left_im)
        
        cv2.waitKey(500); # delay time
    else :
        print 'not found'
      
    print 'find corners...'

############################################################################
  
print 'calibrating...'

objpoints = objpoints[:samples_to_use]
Right_imgpoints = Right_imgpoints[:samples_to_use]
Left_imgpoints = Left_imgpoints[:samples_to_use]

R_ret, R_intrinsic, R_distort, R_rvecs, R_tvecs = cv2.calibrateCamera(objpoints, Right_imgpoints, (width,height), None, None)
print 'Right calibration complete'
print 'R_ret: %s' % R_ret
print 'R_intrinsic: %s' % R_intrinsic
print 'R_distort: %s' % R_distort
print 'R_rvecs: %s' % R_rvecs
print 'R_tvecs: %s' % R_tvecs

L_ret, L_intrinsic, L_distort, L_rvecs, L_tvecs = cv2.calibrateCamera(objpoints, Left_imgpoints, (width,height), None, None)
print 'Left calibration complete'
print 'L_ret: %s' % L_ret
print 'L_intrinsic: %s' % L_intrinsic
print 'L_distort: %s' % L_distort
print 'L_rvecs: %s' % L_rvecs
print 'L_tvecs: %s' % L_tvecs

############################################################################ 

print "Starting Calibration\n"

intrinsic_R2 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)
intrinsic_L2 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)

ret, intrinsic_R2, distort_R2, intrinsic_L2, distort_L2, R, T, E, F = cv2.stereoCalibrate(objpoints, Right_imgpoints, Left_imgpoints, (width, height),criteria=criteria, flags=flags)

print 'intrinsic_R2: %s' % intrinsic_R2
print 'distort_R2: %s' % distort_R2
print 'intrinsic_L2: %s' % intrinsic_L2
print 'distort_L2: %s' % distort_L2
print 'R: %s' % R
print 'T: %s' % T

print "Done Calibration\n"

############################################################################

print "Starting Rectification\n"


Rr, Rl, Pr, Pl, Q, roi1, roi2 = cv2.stereoRectify(intrinsic_R2, distort_R2, intrinsic_L2, distort_L2,(width, height), R, T, alpha=-1)

print 'Rr: %s' % Rr
print 'Rl: %s' % Rl
print 'Pr: %s' % Pr
print 'Pl: %s' % Pl

print "Done Rectification\n"

###########################################################################

print "Applying Undistort\n"

map1x, map1y = cv2.initUndistortRectifyMap(intrinsic_R2, distort_R2, Rr, Pr, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(intrinsic_L2, distort_L2, Rl, Pl, (width, height), cv2.CV_32FC1)

print 'map1x: %s' % map1x
print 'map1y: %s' % map1y
print 'map2x: %s' % map2x
print 'map2y: %s' % map2y

print "Undistort complete\n"

###########################################################################

input_r = "r1.jpg"  # choice that you want to retify image each one~
input_l = "l1.jpg"

input_Rightimg = cv2.imread(input_r)
input_leftimg = cv2.imread(input_l)

right_remap = cv2.remap(input_Rightimg, map1x, map1y, cv2.INTER_NEAREST)
left_remap = cv2.remap(input_leftimg,map2x,map2y,cv2.INTER_NEAREST)

cv2.imshow("input_Rightimg",input_Rightimg);
cv2.imshow("input_leftimg",input_leftimg);

cv2.waitKey(100);

cv2.imshow("image2R", right_remap);
cv2.imshow("image1L", left_remap);
  
cv2.waitKey(100);