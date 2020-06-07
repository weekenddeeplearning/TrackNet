# README for labeling tool

The excutable file is under directory LabelingTool/for_redistribution_files_only, named LabelingTool.exe

You need to check if the MATLAB Runtime is installed before you run the program.

## How to use it?

1. Open the LabelingTool.exe and you will see a window has a main region for image.
2. Click the "Open" button at up-left corner to open the image source, choose and directory for all images. The image will show in main region in the window.
3. Move the cursor on the target and click to label it.
4. The side bar on the right part are the visibility and status of the target:
    * Visibility is divided into 4 parts: No Ball, Easy Identification, Hard Identification, Occluded Ball
    * Status has three circumstances: Flying, Hit, Bouncing
5. Using right arrow key and left arrow key on keyborad can move forward and backward image
6. The text above the main region is noticing user if the image is labeled or not:
    * If it is blue, means already labeled
    * If it is red, means not label yet
7. After labeled all images, click the "Output" button at right down corner to output the labeling file. The labeling file will save as csv file, named "Label.csv" under same directory of images.