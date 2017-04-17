import cv2
import numpy as np

from matplotlib import pyplot as plt



def motion_psf(size, theta):
	"""
	Sets up a convolution/deconvolution kernel based on kernel size and rotation angle

	Inputs:

		size: int  , denotes the size of the kernel
		theta: float, angle of the rotation in degrees. positive values are counter-clockwise
						rotation. Negative values are clockwise.
	
	Outputs:

		kernel: n x n, numpy array of the motion blur kernel. n must be odd 

	"""

	# n needs to be odd
	n = size if (size%2 == 1) else size + 1

	half_n = n//2	 	

	kernel = np.zeros((n,n)).astype('float32')

	# fills in a horizontal line across the middle
	kernel[half_n,half_n:] = 1

	# rotate the kernel to the desired angle
	center = (half_n ,half_n)

	rotation = cv2.getRotationMatrix2D(center, theta, 1.0)

	kernel = cv2.warpAffine(kernel, rotation, (n,n))

	
	#this was to see what the kernel looked like, might have a use for this still
	#plt.imshow(kernel, cmap = 'gray')
	#plt.show()
	

	return kernel

def focus_psf(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    which approximates out of focus blur
    """

    n = l if (l%2 == 1) else l + 1 


    ax = np.arange(-n // 2 + 1., n // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

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
	
	convolved_image = np.zeros_like(proto_convolve, dtype = "float32")

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
			convolved_image[y,x] = conv_val
		

	# lastly we must rescale the image to be values between 0-255
	convolved_image = convolved_image / np.amax(convolved_image)
	convolved_image =(convolved_image * 255).astype('float32')

	image_to_return = convolved_image[pad:image_height+pad, pad:image_width+pad]

	return image_to_return

def deconvolve(convolved_image, PSF):
	"""
	Inputs:

		convolved_image:
					PSF:
				epsilon:
	Outputs:

		deconvolved_image:

	"""

	# create the full sized array with the kernel in it
	#	create the zeros array
	new_kern = np.zeros_like(convolved_image).astype('float32')

	# add in the kernel
	new_kern[:PSF.shape[0], :PSF.shape[1]] = PSF

	plt.imshow(new_kern,cmap='gray')
	plt.title('PADDED PSF')
	plt.show()

	# use the FFT on the convolved image and kernel	
	imag_fft = np.fft.fftn(convolved_image)
	kern_fft = np.fft.fftn(new_kern)
	
	# once in the image is in the FFT we can obtain the formula latent_image = blurred_image / PSF
	# to obtain our deblurred image
	deconvolved_image = np.abs(np.fft.ifftn(imag_fft/kern_fft))
	
	# the image will have shifted during the FFT so we shift it accordingly
	deconvolved_image = np.roll(deconvolved_image, PSF.shape[0]/2, axis = 0)
	deconvolved_image = np.roll(deconvolved_image, PSF.shape[1], axis = 1)

	return deconvolved_image

def add_noise(image,n=300):
	"""
		adds salt and pepper noise to the image

	Inputs:

		image: an NxM numpy array of the image to be blurred
		n 	 : an integer control how much noise is added. The larger n is the less noise there will be

	Outputs:

		noisy_image: an NxM numpy array representing the image with salt and pepper noise added
	"""
	noisy_image = test_conv.copy()

	noise_vec = [(np.random.randint(0,image.shape[0]), np.random.randint(0,image.shape[1])) for k in xrange(image.shape[0] * image.shape[1] / n) ]

	for noise in noise_vec:
		# sets half of the randomly selected locations to 0 and the others to 255 (black and white)
		if np.random.random() < .5:
			noisy_image[noise] = 0
		else:
			noisy_image[noise] = 255

	return noisy_image

#load in the image and display it
path = 'index.jpeg'
image = cv2.imread(path, 0)
#plt.imshow(image, cmap = 'gray')
#plt.show()

# generate our motion kernels
motion_kernel = motion_psf(20,0)
focus_kernel  = focus_psf(11,3)

# show what our kernels look like
#plt.imshow(focus_kernel)
#plt.show()

#plt.imshow(motion_kernel)
#plt.show()

# show what our image looks like convolved with the motion kernel
test_conv  = convolve(image,motion_kernel)
#plt.imshow(test_conv, cmap='gray')
#plt.show()

# show what our image looks like convolved with the focus kernel
deacon = convolve(image, focus_kernel)
#plt.imshow(deacon, cmap = 'gray')
#plt.show()


# show what our image looks like when it's deconvolved with the motion kernel
original2  = np.real(deconvolve(test_conv, motion_kernel))
#plt.imshow(original2, cmap ='gray')
#plt.show()

# show what our image looks like when it's deconvolved with the focus kernel
# this is broken and I don't know why
focus_blur = np.real(deconvolve(deacon, focus_kernel))
#plt.imshow(focus_blur, cmap ='gray')
#plt.show()

new_kern = np.zeros_like(image).astype('float32')

new_kern[:focus_kernel.shape[0], :focus_kernel.shape[1]] = focus_kernel	

#plt.imshow(new_kern, cmap='gray')
#plt.show()

def convolve2(star, psf):
    star_fft = np.fft.fftn(star)
    psf_fft = np.fft.fftn(psf)
    return np.fft.ifftn(star_fft*psf_fft)

def deconvolve2(star, psf):
    star_fft = np.fft.fftn(star)
    psf_fft = np.fft.fftn(psf)
    return np.fft.ifftn(star_fft/psf_fft)


focus_test = convolve2(image,new_kern)
#plt.imshow(np.real(focus_test), cmap = 'gray')
#plt.show()

focus_testd = np.real(deconvolve2(focus_test,new_kern))
#plt.imshow(focus_testd,cmap = 'gray')
#plt.show()


# show what our image looks like when it's deconvolved with the wrong
original2  = np.real(deconvolve(deacon, motion_kernel))
#plt.imshow(original2, cmap ='gray')
#plt.show()

# add some noise to the image
noisy_image = add_noise(focus_test, 400)
cv2.imwrite("noisy_image.jpg",noisy_image)
#plt.imshow(noisy_image, cmap='gray')
#plt.show()


original2  = np.real(deconvolve(noisy_image, motion_kernel))
#plt.imshow(original2, cmap ='gray')
#plt.show()


def variational_calculus(image):
	u = image.copy()
	delta_t = 1e-3
	time_steps = 250
	lam = 1.
	epsilon = 1e-5

	for i in range(time_steps):

	    u_x     = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))/2.
	    u_y     = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0))/2.

	    u_xx    = np.roll(u,-1, axis = 1 ) - 2.*u + np.roll(u, 1 , axis = 1)
	    u_yy    = np.roll(u,-1, axis = 0 ) - 2.*u + np.roll(u, 1 , axis = 0)
	    
	    u_xy    = (np.roll(u_x, -1, axis=0) - np.roll(u_x, 1, axis=0))/2.
	    
	    u_t     = -lam*(u - image) + (u_xx*u_y**2 + u_yy*u_x**2 - 2*u_x*u_y*u_xy) / (epsilon + u_x**2 + u_y**2)**(3./2)
	    
	    u_next  = u_t * delta_t + u
	    u       = u_next
	    
	plt.imshow(u , cmap = 'gray')
	plt.show()

	return u

def variational_calculus(image):
	u = image.copy()
	delta_t = 1e-3
	time_steps = 250
	lam = 1.
	epsilon = 1e-5

	for i in range(time_steps):

	    u_x     = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))/2.
	    u_y     = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0))/2.

	    u_xx    = np.roll(u,-1, axis = 1 ) - 2.*u + np.roll(u, 1 , axis = 1)
	    u_yy    = np.roll(u,-1, axis = 0 ) - 2.*u + np.roll(u, 1 , axis = 0)
	    
	    u_xy    = (np.roll(u_x, -1, axis=0) - np.roll(u_x, 1, axis=0))/2.
	    
	    u_t     = -lam*(u - image) + (u_xx*u_y**2 + u_yy*u_x**2 - 2*u_x*u_y*u_xy) / (epsilon + u_x**2 + u_y**2)**(3./2)
	    
	    u_next  = u_t * delta_t + u
	    u       = u_next
	    
	plt.imshow(u , cmap = 'gray')
	plt.show()

	return u


variation = variational_calculus(noisy_image)

cv2.imwrite("variation.jpg",variation)

dif = noisy_image - variation

plt.imshow(dif,cmap='gray')
plt.show()



"""
plt.show(deconv,cmap='gray')
plt.show()

dif = test2_conv - test2_conv
print dif.shape
zeros = np.zeros_like(dif)
print np.allclose(dif,zeros)

plt.imshow(dif, cmap = 'gray')
plt.show()


restored = deconvolve(test_conv,kernel)
plt.imshow(restored, cmap = 'gray')
plt.show()
""" 