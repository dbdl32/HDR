import cv2
import numpy as np
import rawpy
from matplotlib import pyplot as plt
# def display(img):
# 	plt.imshow(img)
# 	# to hide tick values on X and Y axis
# 	plt.xticks([]), plt.yticks([])  
# 	plt.show()

def tau(imgR,imgS):
	color = ('b','g','r')
	height = len(imgR)
	width = len(imgR[0])
	res = np.zeros((height,width,3), np.uint8)
	for k,col in enumerate(color):
		# histr,binr = np.histogram(imgR.flatten(),256,[0,256])
		# cdfr = histr.cumsum()
		# cdfr_normalized = cdfr * histr.max()/ cdfr.max()
		# hists,bins = np.histogram(imgS.flatten(),256,[0,256])
		# cdfs = hists.cumsum()
		# cdfs_normalized = cdfs * hists.max()/ cdfs.max()
		# get statistic histogram for current channel for both image
		histr = cv2.calcHist([imgR],[k],None,[256],[0,256])
		hists = cv2.calcHist([imgS],[k],None,[256],[0,256])
		# normalize both images
		# histr_norm = cv2.normalize(histr,1)
		# hists_norm = cv2.normalize(hists,1)
		n = height*width
		n*=1.0
		histr_norm = []
		for i in range(0,256):
			histr_norm.append(histr[i][0]/n)
		#print(histr_norm)
		n = len(imgS)*len(imgS[0])
		n*=1.0
		hists_norm = []
		for i in range(0,256):
			hists_norm.append(hists[i][0]/n)
		# get accumulative histogram for both image
		sum1 = 0
		sum2 = 0
		histr_C = []
		hists_C = []
		for i in range(0,256):
			sum1 += histr_norm[i]
			sum2 += hists_norm[i]
			histr_C.append(sum1)
			hists_C.append(sum2)
		#print(hists_C)
		# get look up table
		lookup = []
		PG = 0
		for i in range(0,256):
			min_val = 1000.0
			for j in range(0,256):
				if (hists_C[j]-histr_C[i]) < min_val and (hists_C[j]-histr_C[i]) >= 0:
					min_val = hists_C[j]-histr_C[i]
					PG = j
			lookup.append(PG)
		#print('------------------------------------')
		#print(lookup)
		# image transfer
		for i in range(0,height):
			p = imgR[i,:,k]
			for j in range(0,width): 
				val = p[j]
				res.itemset((i,j,k),lookup[val])
	return res

# raw = rawpy.imread('pics/1_1600.NEF')
# imgR = raw.postprocess()
# #imgR = cv2.cvtColor(imgR,cv2.COLOR_RGB2BGR)
# #imgR = cv2.imread('1_result.jpg')
# raw = rawpy.imread('pics/1_100.NEF')
# imgS = raw.postprocess()
# res = tau(imgR,imgS)
# #cv2.imwrite('tau.png',res)
# display(res)