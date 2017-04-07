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

	kernel = np.zeros((n,n)).astype('float32')

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
	kernel_height, kernel_width = kernel.shape

	# pad the image 
	pad = (kernel_width -1) /2
	
	proto_convolve = cv2.copyMakeBorder(image, pad, pad, pad, pad,
										    cv2.BORDER_REPLICATE)

	convolved_image = np.zeros((image_height, image_width), dtype = "float32")

	# loop over the proto convolution, while sliding our kernel
	# We're going left to right then top to bottom


	# Top to bottom
	for y in xrange(pad, image_height + pad):

		# Left to right
		for x in range(pad, image_width + pad):

			# extract a region of interest(roi) to convolve
			roi = proto_convolve[ y-pad:y+pad+1, x-pad:x+pad+1]

			# perform the convolution using the kernel
			# Element wise multiply and then sum the submatrix
			conv_val = (roi * kernel).sum()
		
			# store the value in the output in the convolved image output
			convolved_image[y-pad,x-pad] = conv_val
		

	# lastly we must rescale the image to be values between 0-255
	convolved_image = convolved_image / np.amax(convolved_image)
	convolved_image =(convolved_image * 255).astype('float32')


	return convolved_image

def deconvolve(convolved_image, PSF, epsilon = 1e-4):
	"""
	Inputs:

		convolved_image:
					PSF:
				epsilon:


	Outputs:

		deconvolved_image:

	"""
	height, width = convolved_image.shape 

	# in order for us to center the kernel we must have an array with odd dimensions
	
	# assume the that the array is odd if not set the padding to 1
	h_pad = 0 if height%2 == 1 else 1
	w_pad = 0 if  width%2 == 1 else 1

	# pad on top and bottom
	padded_convolve = cv2.copyMakeBorder(convolved_image,h_pad,0,w_pad,0, cv2.BORDER_REPLICATE)

	# use the FFT to ppn the convolved image	
	imag_fft = np.fft.fftn(padded_convolve)

	# create the full sized array with the kernel centered in it
	#	create the zeros array
	new_kern = np.zeros_like(padded_convolve)

	# center the kernel in the array
	h_shift = (height - PSF.shape[0])/2 
	w_shift = (width  - PSF.shape[1])/2 

	new_kern[h_shift:h_shift+PSF.shape[0], w_shift:w_shift+PSF.shape[1]] += PSF/PSF.sum() + epsilon

	plt.imshow(new_kern)
	plt.show()

	kern_fft = np.fft.fftn(new_kern)
	print kern_fft,imag_fft
	deconvol = np.divide(imag_fft,kern_fft)

	deconvolved_image = np.abs(np.fft.ifftn(deconvol))
	
	return deconvolved_image	



path = 'index.jpeg'
image = cv2.imread(path, 0)
#plt.imshow(image, cmap = 'gray')
#plt.show()

kernel = motion_psf(20,0)

test_conv = convolve(image,kernel)
#plt.imshow(test_conv, cmap='gray')
#plt.show()

restored = deconvolve(test_conv,kernel)
plt.imshow(restored.reshape(restored.shape[0],restored.shape[1]), cmap = 'gray')
plt.show()