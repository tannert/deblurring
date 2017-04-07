import cv2

import numpy as np

from matplotlib import pyplot as plt


def motion_psf(size, theta):
	"""
	Inputs:

		 size: int  , denotes the size of the kernel
		theta: float, angle of the rotation in degrees. positive values are counter-clockwise
						rotation. Negative values are clockwise.
	
	Outputs:

		kernel: n x n, numpy array of the motion blur kernel. n must be odd 

	"""

	
	if size%2 == 1:
		# leave it as is if already odd
		n = size

	else:
		# if it's even then make it odd
		n = size + 1

	half_n = int(np.floor(n/2))	 	

	kernel = np.zeros((n,n))

	# fills in a horizontal line across the middle
	kernel[half_n,:] = 1

	# rotate the kernel to the desired angle
	center = (half_n ,half_n)

	rotation = cv2.getRotationMatrix2D(center, theta, 1.0)

	kernel = cv2.warpAffine(kernel, rotation, (n,n))

	"""
	 this was to see what the kernel looked like, might have a use for this still
	plt.imshow(kernel, cmap = 'gray')
	plt.show()
	"""
	return kernel

def convolve(image, kernel):
	"""
	Inputs:

		 image: a NxM array,	A grayscale image represented as an numpy array
		kernel:	a KxK array,	A kernel that we will convovlve with our image.
					 			K is odd. If it isn't someone did something very wrong
	
	Outputs:

		convolved_image: a PxP array, 	Our padded image array that's been convolved with 
										the kernel.

	"""

	# Save the spatial dimensions of the image and the kernel
	image_height, image_width   = image.shape
	kernel_height, kernel_width = kenrel.shape

	# pad the image 
	pad = (kernel_width -1) /2

	proto_convolve = cv2.copyMakeBorder(image, pad, pad, pad, pad)

	convovlve_imge = np.zeros((image_height, image_width), dtype = "float32")

	"""
	TO BE BUILT:

	NESTED FOR LOOPS FOR MOVING CONVOLUTION

	RETURN STATEMENTS 
	"""