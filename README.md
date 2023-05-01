# <center>Multi-Auto-Annotate</center>

---

This tool is intended to annotate given data label(s) in all images in a given directory. This is useful and handy for ***Object Identification*** and ***Computer Vision*** purposes.

All it takes is one single command in your terminal and then you can just sit back and watch the segmentation and labelling happen automatically.

You also have the option to review it as it happens by using the `--displayMaskedImage=True` argument in your command.

You can use the open [COCO dataset](https://cocodataset.org/) to annotate common objects in your images without having to train a model yourself.

This tool is built on top of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and forked from the very useful and much appreciated repository [Auto-Annotate](https://github.com/mdhmz1/Auto-Annotate) by [Muhamman Hamzah](https://github.com/mdhmz1).

This tool works in two modes -

1. COCO Label Annotation - *No training required*. Uses the pre-trained weights of the the COCO dataset. Point to the directory using the `--image-directory=<directory_path>` argument and the annotations will be ready in a while.
2. Custom Label Annotation - Train the model for custom labels  and use the trained weights for auto-annotation.

*No known issues have been encountered by me till now, but feel free to raise an issue if you come across one while using the program.* (The issues that I encountered while using the program in original repository have been fixed by me as as far as I know.)

## Annotations Format

The Annotations are stored in a JSON format with all the relevant details in the following format -

### JSON Format : -

```
{
  "filename": "image_name",
  "objects": [
    {
      "id": 1,
      "label": "label_1",
      "bbox": [ x, y, w, h ], -- x,y coordinate of top left point of bounding box
                            -- w,h width and height of the bounding box
      "segmentation": [
      [ x1, y1, x2, y2,...] -- For X,Y belonging to the pixel location of the mask segment
      ]
    },
      {
      "id": 2,
      "label": "label_2",
      "bbox": [x, y, w, h ], -- x,y coordinate of top left point of bounding box
                            -- w,h width and height of the bounding box
      "segmentation": [
      [ x1, y1, x2, y2,...] -- For X,Y belonging to the pixel location of the mask segment
      ]
    }
  ]
}
```

### Sample JSON : -

```
{
  "filename": "dgct1.jpg",
  "objects": [
    {
      "id": 1,
      "label": "dog",
      "bbox": [ 93.5, 15.5, 149, 162],
      "segmentation": [
        [224, 177.5, 217, 177.5, 203.5, 168, 200.5, 151, 195, 143.5, 193, 143.5, 186.5, 151, 185.5, 159, 182.5, 164, 175, 167.5, 163, 169.5, 149, 168.5, 134, 161.5, 130, 161.5, 119, 166.5, 111, 166.5, 108.5, 164, 108.5, 158, 122, 144.5, 128, 143.5, 132, 145.5, 136.5, 141, 136.5, 106, 134.5, 99, 127, 91.5, 122, 90.5, 114, 85.5, 99, 83.5, 93.5, 75, 95.5, 65, 101.5, 53, 102.5, 44, 107.5, 33, 127, 15.5, 151, 15.5, 173, 26.5, 179.5, 33, 186.5, 49, 207.5, 68, 209.5, 72, 213.5, 75, 219.5, 86, 235.5, 104, 237.5, 111, 241.5, 117, 242.5, 144, 241.5, 150, 236.5, 157, 229.5, 174, 224, 177.5]
      ]
    },
    {
      "id": 2,
      "label": "dog",
      "bbox": [14.5, 85.5, 73, 88],
      "segmentation": [
        [72, 173.5, 46, 173.5, 33, 170.5, 27.5, 165, 24.5, 156, 14.5, 148, 17, 143.5, 28, 142.5, 33, 139.5, 37.5, 134, 40.5, 127, 41.5, 100, 48, 90.5, 61, 85.5, 73, 85.5, 78, 87.5, 82.5, 91, 85.5, 98, 85.5, 107, 82.5, 114, 82.5, 121, 86.5, 128, 87.5, 146, 79.5, 169, 72, 173.5]
      ]
    }
  ]
}
```

## Installation

1. Clone this repository.
2. Install dependencies.
	```
	pip install -r requirements.txt
	```
3. If planning to use pre-trained COCO weights, download the weights file trained on COCO dataset from Mask R-CNN repository.
	- [Mask R-CNN Releases](https://github.com/matterport/Mask_RCNN/releases): Check for the new file here. It should be named `mask_rcnn_coco.h5`.The weightes I used are from Mask R-CNN 2.0.
	
4. If planning to train your own model for objects not in the COCO dataset, train Mask-RCNN accordingly and use those weights instead with the `--weights` argument in the execution command.
5. Installation complete!

## One Command to Annotate them All

You'll have to give a different kind of command depending upon whether you're using COCO weights or not.

### For Multiple Labels Annotation -

***You'll have to configure labels in the `multi-annotate.py` file as described in the next section for it to work.*** <br>
The *default labels set by me are "cat" and "dog"*, so unless you want your images to be segmented for only kawaii neko-chans and inu-chans, please read the next section and change the labels.

```
python multi-annotate.py annotateCoco --image_directory=/path_to_the_image_directory/ --labels=True
```

If you're using cutom trained weights, use this command instead -

```
python multi-annotate.py annotateCustom --image_directory=/path_to_the_image_directory/ --weights=/path_to/weights.h5 --labels=True
```

If you want to see and save the masked versions of the segmented images, use
```
--displayMaskedImages=True
```
argument and you'll be able to review things as they happen. You'll need to close the image viewer's window each time for the program to move ahead though.


### For Single Label Annotation -

```
python multi-annotate.py annotateCoco --image_directory=/path_to_the_image_directory/ --label=single_label_from_COCO
```

If you're using cutom trained weights, use this command instead -

```
python multi-annotate.py annotateCustom --image_directory=/path_to_the_image_directory/ --weights=/path_to/weights.h5 --label=single_label_from_trained_weights
```

If you want to see and save the masked versions of the segmented images, use
```
--displayMaskedImages=True
```
argument and you'll be able to review things as they happen. You'll need to close the image viewer's window each time for the program to move ahead though.

## Configure Labels for Automated Multi Annotation !important

You need to configure the program file for it to work with multiple labels. Follow the steps below -

1. Open `multi-annotate.py` in any IDE or Text Editor.
2. Use CTRL+F to find the `set_labels_here` list.
3. Enter the labels in list format, with each item as a string. <br>
	e.g.
	```
	set_labels_here = ['cat', 'dog', 'tv']
	```
	*Note that the labels entered here should have trained weights provided for them or the program will fail. Same for the single label passed in the command argument.*
4. Save the file.

## All Done!

By now you should be ready to automatically annotate and label bulk of images in the whole directory. <br>

***Star this repo and raise issues if you face any.***

If you want to figure out how to train a model on your own dataset, check out the original blog post [about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) by Waleed Abdulla where he explained the process starting from annotating images to training to using the results in a sample application.

The use train.py which is a modified version of balloon.py written by Waleed to support only the training part. Here are the commands for that -

```
    # Train a new model starting from pre-trained COCO weights
    python3 customTrain.py train --dataset=/path/to/custom/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 customTrain.py train --dataset=/path/to/custom/dataset --weights=last
```

I've not checked and touched this part of the code from the [original repository](https://github.com/mdhmz1/Auto-Annotate), and will do so and smooth out any kinks and issues I face when I do. Feel free to raise issues meanwhile.

---

### <center> All the best in your projects and adventures! </center>

---