# TrackNet
Heatmap based high speed tiny sport objects tracking based on TrackNet

![TrackNet](/TrackNet.gif)


## Usage

### Installation

```bash
pip install -r requirements.txt
```


### Inference

```bash
# video file with output
cd Code_Baddy
python predict_video.py  --save_weights_path=weights/model_baddy.h5 --input_video_path=path_to_file.mp4 --output_video_path="tracked.mp4" --n_classes=256
```

## Labeling Tool: Make your own dataset
See the readme file in the LabelingTool directory

### Training

1. Use the TrackNet_Python.ipynb to create heatmap as Ground Truth, and save heatmap as JPG file.  (First Cell)
	 
2. Create the training and testing csv files (Second cell)
	 Change the folder path to your custom paths for 1. and 2.
3. Copy the training.csv and testing.csv file to Code_Baddy/Code_Custom folder
4. After have training images and ground truth, we can start to train the TrackNet model II (Three frame Input)
	* Open command line
	* Change directory to Code_Baddy/Code_Custom
	* Using following command as example, you may need to change the command:

```bash
# Train with custom dataset
python train.py --save_weights_path=weights/model.h5 --training_images_name="training.csv" --epochs=500 --n_classes=256 --  input_height=360 --input_width=640 --load_weights=2 --step_per_epochs=200 --batch_size=2
```

    * Trained model weight will be save in weights/model.h5
    
	* Detailed explanation
			--save_weights_path: Save the weight path
			--training_images_name: Training images csv file path, training.csv
			--epochs: Epochs be set as 500 in this work
			--n_classes: Last layer classes, since the output value of TrackNet is between 0-255, the last layer depth be set as 256 
			--input_height: Input height be resize as 360 in this work
			--input_width: Input width be resize as 640 in this work
			--load_weights: If you want to retrain the weights from previous weight, give the number of weight in weights/model. If not, delete it.
			--step_per_epochs: Step per Epochs be set as 200 in this work
			-batch_size: Batch size be set as 2 in this work

## References

GitHub : [TrackNet](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet/tree/master)

Paper  : [TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications](https://ieeexplore.ieee.org/abstract/document/8909871/authors#authors)
