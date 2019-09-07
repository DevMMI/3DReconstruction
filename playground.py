import numpy as np
import cv2 as cv
from numpy.linalg import inv
# img = cv.imread('/home/mohamedisse/Documents/ComputerVision/reconstruction/samples/stereo_front_left.jpg')
# gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg',img)

## computing R and T from essential matrix ##
# essential_mat = np.asarray([[-1.77513463076033E-07, 7.96021510200727E-06, -0.0063605768],[-8.31942693313702E-06, 1.72182922070321E-06, 0.3958693675], [0.0155116686, -0.398708109, 1] ])
#
# u, s, vh = np.linalg.svd(essential_mat, full_matrices=True)
#
# W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
# Z = np.asarray([[0, 1, 0],[-1, 0, 0], [0, 0, 0]])
#
# # first method
# tx = u * W * s * np.transpose(u)
# r = u * inv(W) * vh
# np.savetxt("math/tx.csv", tx, delimiter=",")
# np.savetxt("math/rot.csv", r, delimiter=",")
#
# # first method
# tx_alt = u * Z * np.transpose(u)
#
# np.savetxt("math/tx_alt.csv", tx_alt, delimiter=",")


right = np.asarray([ [ -0.0096531,  0.0002185,  0.9999534 ], [  -0.9999002,  0.0103117, -0.0096548 ], [  -0.0103133, -0.9999468,  0.0001190 ] ] )

left = np.asarray([ [  0.0017554, -0.0020406,  0.9999964 ], [  -0.9999327, -0.0114710,  0.0017319 ], [   0.0114674, -0.9999321, -0.0020606 ] ] )




quat_diff_secondary = left * inv(right)

quat_diff_likely = right * inv(left)

f = open("quat_likely.csv", "ab")
f_sec = open("quat_secondary.csv", "ab")

np.savetxt(f, quat_diff_likely, delimiter=",")
np.savetxt(f_sec, quat_diff_secondary, delimiter=",")
