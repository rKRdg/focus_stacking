import cv2
import numpy as np
import glob
import math
import config



def align_image(image1,image2):
	print("alignment starts...")
	if config.ALIGNMENT_DETECTOR=="SIFT":
		print(f"use SIFT detector with {config.ALIGNMENT_FEATURE_NUM} features")
		detector = cv2.xfeatures2d.SIFT_create(config.ALIGNMENT_FEATURE_NUM)
	elif config.ALIGNMENT_DETECTOR=="SURF":
		print(f"use SURF detector with {config.ALIGNMENT_FEATURE_NUM} features")
		detector = cv2.xfeatures2d.SURF_create(config.ALIGNMENT_FEATURE_NUM)

	kp1, des1 = detector.detectAndCompute(image1, None)
	kp2, des2 = detector.detectAndCompute(image2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)


	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(m)


	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	

	imgout=cv2.warpPerspective(image2, np.linalg.inv(M), (image2.shape[1], image2.shape[0]))

	print("alignment finished...")

	return imgout




def laplacian_trans(image):
	kernel_size = config.LAP_KERNEL_SIZE        # Size of the laplacian window
	blur_size = config.LAP_KERNEL_SIZE         #  gaussian blur kernel size (odd)

	blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
	
	return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


def deviation(xmin,ymin,w,img):
	m, s = cv2.meanStdDev(img[xmin:xmin+w,ymin:ymin+w])
	return np.mean(s)


def hist_match(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)


def focus_stack(image1, image2):
	print("start focus stacking")

	if config.HIST_MATCHING:
		img_yuv0 = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
		img_yuv1 = cv2.cvtColor(image2, cv2.COLOR_BGR2YUV)


		img_yuv1[:,:,0] = hist_match(img_yuv1[:,:,0],img_yuv0[:,:,0])
		image2 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)

	
	tmplap = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
	tmplap = laplacian_trans(tmplap)
	kernel = np.ones((5,5),np.float32)
	tmplap = cv2.filter2D(tmplap,-1,kernel)
	lap1 = np.absolute(tmplap)


	tmplap = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
	tmplap = laplacian_trans(tmplap)
	kernel = np.ones((5,5),np.float32)
	tmplap = cv2.filter2D(tmplap,-1,kernel)
	lap2 = np.absolute(tmplap)



	output = np.zeros(shape=image1.shape, dtype="float64")
	output_count = np.zeros(shape=image1.shape, dtype="int")
	

	
	

	w=config.SLIDE_WINDOW_SIZE
	step=config.SLIDE_WINDOW_STEP
	
	
	x,y=0,0
	while x<lap1.shape[0]:
		while y<lap1.shape[1]:
			dev1=deviation(x,y,w,lap1)
			dev2=deviation(x,y,w,lap2)
			if dev1>dev2:
				output[x:x+w,y:y+w,:]+=image1[x:x+w,y:y+w,:]
				output_count[x:x+w,y:y+w,:]+=1
			elif dev2>dev1:
				output[x:x+w,y:y+w,:]+=image2[x:x+w,y:y+w,:]
				output_count[x:x+w,y:y+w,:]+=1
			else:
				tmpmean=image1[x:x+w,y:y+w,:]*0.5+image2[x:x+w,y:y+w,:]*0.5
				output[x:x+w,y:y+w,:]+=tmpmean
				output_count[x:x+w,y:y+w,:]+=1


			y+=step
		y=0
		x+=step
		
	output=output/output_count

	print("finished")

	return output.astype(np.uint8)



images=[]
print("loading files...")
baseimage=cv2.imread(config.IMAGE_PATH+config.BASE_IMAGE)
print("Base: "+config.IMAGE_PATH+config.BASE_IMAGE)
for name in glob.glob(config.IMAGE_PATH+"*.*"):
	if name==config.IMAGE_PATH+config.BASE_IMAGE:
		continue
	images.append(cv2.imread(name))
	print(name)
print("loading finished")

image1=baseimage
image2=align_image(baseimage,images[0])


result=focus_stack(image1,image2)
if config.DENOISE:
	result=cv2.fastNlMeansDenoisingColored(result,None,h = 3,hColor = 3,templateWindowSize = 7,searchWindowSize = 21 )

cv2.imwrite("result.png", result) 
