# imported libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
 
# set rows and cols
nrows = 2
ncols = 3

# original image but in BRG
img = cv2.imread('goldenGate.jpg') 

# original image in grey
gray = cv2.imread('goldenGate.jpg',0) #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# original image in RGB
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(gray, cmap = plt.get_cmap('gray'))

# plot and dispaly images
plt.figure(1)

#Image 1 - Original image
plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray') 
plt.title('Original'), plt.xticks([]), plt.yticks([]) 

#Image 2 - Gray Scale image
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray') 
plt.title('Grey Scale'), plt.xticks([]), plt.yticks([]) 

# Image 3
# variables for corner detection
blockSize = 2;
aperture_size = 3;
k = 0.04;

# list of all conners
dst = cv2.cornerHarris(gray, blockSize, aperture_size, k)

# deep copy
imgHarris = img.copy()

# variables for Harris Corner
threshold = 0.1; #number between 0 and 1
B = 100;
G = 0;
R = 255;

# loop through the 2d matrix - dst 
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

# plot image 3 - Harris Corners     
plt.subplot(nrows, ncols,3),plt.imshow(imgHarris, cmap = 'gray') 
plt.title('Harris Corners'), plt.xticks([]), plt.yticks([]) 

# Image 4
# variables for corner detection - Shi Tomasi algorithm
maxCorners = 50
qualityLevel = 0.01
minDistance = 10

# Shi Tomasi algorithm (also known as Good Features to Track (GFTT))
corners = cv2.goodFeaturesToTrack(gray,maxCorners,qualityLevel,minDistance)

# deep copy
imgShiTomasi = img.copy()

# variables for corner detection - Shi Tomasi
B = 100;
G = 0;
R = 255;

# loop through the corners array
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(B, G, R),-1)

# image 4 - Shi Tomasi algorithm     
plt.subplot(nrows, ncols,4),plt.imshow(imgShiTomasi, cmap = 'gray') 
plt.title('Shi Tomasi algorithm'), plt.xticks([]), plt.yticks([]) 

# deep copy of img
imgSift = img.copy();

#Initiate SIFT detector 
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

#Draw keypoints  
imgSift = cv2.drawKeypoints(imgSift,kps,outImage=None,color=(B,G,R),flags=4)


# image 5 - imgSift     
plt.subplot(nrows, ncols,5),plt.imshow(imgSift, cmap = 'gray') 
plt.title('Sift Dector'), plt.xticks([]), plt.yticks([]) 

#=========================================================

# deep copy of img
imgSift50 = img.copy();

##Initiate SIFT detector with a max of 50 keypoints
maxKps = 50
sift = cv2.xfeatures2d.SIFT_create(maxKps)
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

#Draw keypoints  
imgSift50 = cv2.drawKeypoints(imgSift50,kps,outImage=None,color=(B,G,R),flags=4)

# image 6 - imgSift50     
plt.subplot(nrows, ncols,6),plt.imshow(imgSift50, cmap = 'gray') 
plt.title('Sift Dector 50 kps'), plt.xticks([]), plt.yticks([]) 


# image1 = cv2.imread('GMIT1.jpg',0) # queryImage
# image2 = cv2.imread('GMIT2.jpg',0) # trainImage

# # Initiate SIFT detector
# orb = cv2.ORB_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(image1,None)
# kp2, des2 = orb.detectAndCompute(image2,None)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.
# image3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[:10],None, flags=2)

# plt.subplot(nrows, ncols,6),plt.imshow(image3, cmap = 'gray')
# plt.title('ORB'), plt.xticks([]), plt.yticks([]) 


plt.show() 