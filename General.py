'''
@ author Bean, Fangte, Yike,
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import rawpy
from patchmatch import *
from tau import *
from hdr_fusion import *

# Read raw images. Here we use 1_320 as the reference image. Typically, an image under moderate 
# exposure time has the most well exposued pixels.
raw = rawpy.imread('1_320.NEF')
imgR = raw.postprocess()
imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2BGR)
imgR = cv2.resize(imgR,None,fx=0.1,fy=0.1)
cv2.imwrite('imgR.png', imgR)

# Read two other images as source images
raw1 = rawpy.imread('2_800.NEF')
imgS1 = raw1.postprocess()
imgS1 = cv2.cvtColor(imgS1, cv2.COLOR_RGB2BGR)
imgS1 = cv2.resize(imgS1,None,fx=0.1,fy=0.1)
cv2.imwrite('imgS1.png', imgS1)

raw2 = rawpy.imread('3_160.NEF')
imgS2 = raw2.postprocess()
imgS2 = cv2.cvtColor(imgS2, cv2.COLOR_RGB2BGR)
imgS2 = cv2.resize(imgS2,None,fx=0.1,fy=0.1)
cv2.imwrite('imgS2.png', imgS2)

# Use HDR imaging method provided by openCV. You can observe severe ghosting in imgHDR.  
stack = [imgR, imgS1, imgS2]
# combine() is a function in hdr_fusion.py with HDR method in openCV.  
imgHDR = combine(stack)
cv2.imwrite('imgHDR.png', imgHDR)

# tau() is a function in tau.py. This is a color mapping function to transfer the reference
# image to a latent image with the same exposure condition as the source image. Here, you can
# observe the brightness of imgL1 is samiliar to the brightness of imgS1. But the scene in
# imgL1 is just the scene in imgR.
imgL1 = tau(imgR, imgS1)
cv2.imwrite('imgL1.png', imgL1)

imgL2 = tau(imgR, imgS2)
cv2.imwrite('imgL2.png', imgL2)

# Here, we can generate a HDR image just based on the color-mapped latent images as well as the
# reference image.
stack1 = [imgR, imgL1, imgL2]
imgHDR1 = combine(stack1)
cv2.imwrite('imgHDR1.png', imgHDR1)

# NNF is a function in patchmatch.py. You can change the boxsize in PatchMatch (the match window).
# nnf1.primeReconstruct() can reconstruct a match image based on the matched patches in 
# imgS1. Here, only the location of these patches are moved to the same location in imgL1
nnf1 = NNF(imgL1, imgS1, boxsize=10) 
nnf1.solve()
imgL1s = nnf1.primeReconstruct()
cv2.imwrite('imgL1s.png', imgL1s)

nnf2 = NNF(imgL2, imgS2, boxsize=10) 
nnf2.solve()
imgL2s = nnf2.primeReconstruct()
cv2.imwrite('imgL2s.png', imgL2s)

# Here, if the pixels in imgR are too dark or too too bright, we change the pixels in the mapped 
# images to the pixels in the PatchMatch reconstructed images.
for i in range(0, imgR.shape[0],1):
	for j in range(0, imgR.shape[1],1):
			if (imgR[i][j][0] > 235 and imgR[i][j][1] > 235 and imgR[i][j][2] > 235)  or (
				imgR[i][j][0] < 20 and imgR[i][j][1] < 20 and imgR[i][j][2] < 20):
				imgL1[i][j][0] = imgL1s[i][j][0] 
				imgL1[i][j][1] = imgL1s[i][j][1] 
				imgL1[i][j][2] = imgL1s[i][j][2] 



# The final latent images are optimized based on color-mapped images and PatchMatch reonstructed
# images. 
imgL1ss = (imgL1 * 0.8 + imgL1s * 0.2).astype(np.uint8)
cv2.imwrite('imgL1ss.png', imgL1ss)
imgL2ss = (imgL2 * 0.8 + imgL2s * 0.2).astype(np.uint8)
cv2.imwrite('imgL2ss.png', imgL2ss)
# Final HDR imaging process based on our results.
stackOurMethod = [imgR, imgL1ss, imgL2ss]
imgOurMethod = combine(stackOurMethod)
cv2.imwrite('imgOurMethod.png', imgOurMethod)

