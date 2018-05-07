###################### Section 2.1 import all the dependencies ###############################
#import opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from PIL import Image, ImageFilter
from scipy import ndimage, misc
##################################################################


def combine(img_stack):
	alignMTB = cv2.createAlignMTB()
	alignMTB.process(img_stack, img_stack)
	#exposure_times = np.array([1/100, 1/160, 1/320, 1/500,1/800,1/1600], dtype=np.float32)
	#do HDR calculation
	 # Merge exposures to HDR image
	merge_mertens = cv2.createMergeMertens()
	res_mertens = merge_mertens.process(img_stack)
	res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
	################################################################
	#imgBGR = cv2.cvtColor(res_mertens_8bit,cv2.COLOR_RGB2BGR)
	return res_mertens_8bit