import argparse
import Models , LoadBatches
import cv2
import numpy as np
import glob
import os

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--test_images_path", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--output_height", type=int , default = 224  )
parser.add_argument("--output_width", type=int , default = 224 )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()
n_classes = args.n_classes
images_path = args.test_images_path
output_path = args.output_path
input_width =  args.input_width
input_height = args.input_height
output_width =  args.output_width
output_height = args.output_height

#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN( n_classes , input_height=input_height, input_width=input_width )
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta', metrics=['accuracy'])
m.load_weights( args.save_weights_path )


#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

#create grayscale RGB (0,0,0)~(255,255,255)
colors = [  ( i, i, i  ) for i in range(0, n_classes)  ]

#predict each clips from 1 to 82
for clip in range(1,82):

	#get all JPG images in the path
	images = glob.glob( images_path + str(clip) + "/*.jpg" )
	images.sort()


	#create folder for saving output image  
	if not os.path.exists(output_path + str(clip) + "/"):
	    os.makedirs(output_path + str(clip) + "/")

	#predict each images
	#since TrackNet cant predict first and second images, so we start from third image
	for i in range(2,len(images)):

		output_name = images[i].replace( images_path,  output_path)

		#load input data
		X = LoadBatches.getInputArr( images[i], images[i-1], images[i-2], input_width, input_height )
		#prdict heatmap
		pr = m.predict( np.array([X]) )[0]

		#since TrackNet output is ( net_output_height*model_output_width , n_classes )
		#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
		pr = pr.reshape(( model_output_height, model_output_width , n_classes ) ).argmax( axis=2 )
		
		#declare variable for output image
		output_img = np.zeros( (  model_output_height, model_output_width , 3  ) )

		#set RGB value for each pixel predciton
		for c in range(n_classes):
			output_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
			output_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
			output_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

		#reshape the image size as original input image
		output_img = cv2.resize(output_img  , (output_width , output_height ))

		#output heatmap image
		cv2.imwrite(  output_name , output_img )
