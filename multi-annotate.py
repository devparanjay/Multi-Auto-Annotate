import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import json
import os
from pathlib import Path
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from mrcnn import utils
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#tf.logging.set_verbosity(tf.logging.ERROR)

set_labels_here = ['cat', 'dog']

def annotateResult(result, image_name, image_path, label):
    n = len(result['class_ids'])
    #annotations = []
    #annotations = {
    #    'filename': image_name,
    #    'objects': []
    #}
    json_name = image_name.split('.')[0] + '.json'
    #print(json_name)
    json_path = image_path + json_name
    json_file = Path(json_path)
    # print(json_path)
    #annotations = {}
    annotationId = 1
    #if not os.path.isfile(json_path):
    #    print("!!! JSON Path not Detected! !!!")
    if os.path.isfile(json_path):
    #if json_file.is_file:
        with open(json_path, 'r') as img_json_file:
            img_json = img_json_file.read()
            annotations = json.loads(img_json)
            annotationId = annotations['objects'][len(annotations['objects'])-1]['id'] + 1
            print("  - Existing annotations found, I will add the current annotation to them.")
            print("  - Current Annotation Number:", annotationId)
        #print(annotations)
    else:
        print('  - No annotations file found; I will make a new file called "' + json_name + '" in the image directory.\n')
        annotations = {
        'filename': image_name,
        'objects': []
    }
    annotations['filename'] = image_name

    for i in range(n):
        #annotationId = i+1
        if class_names[result['class_ids'][i]] == label:
            annotation = create_sub_mask_annotation(result['masks'][:, :, i], result['rois'][i],
                                                annotationId, result['class_ids'][i], image_name)
            #annotations['objects'] = annotations['objects'] + annotation
            #annotations['objects'] = 'is the code here working even'


            annotations['objects'].append(annotation)
            annotationId += 1
    return annotations


def create_sub_mask_annotation(sub_mask, bounding_box, annotationId, classId, image_name):
    # Find contours (boundary lines) around each sub-mask
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    # area = multi_poly.area

    annotation = {'id': annotationId,'label': str(class_names[classId]),'bbox': bbox,'segmentation': segmentations,}
#    annotated_object = {
#        'id': annotationId,
#        'label': str(class_names[classId]),
#        'bbox': bbox,
#        'segmentation': segmentations,
#    }
#    for l in range(len(labels)):

    return annotation



def writeToJSONFile(path, fileName, data):
    fileName = fileName.split(".")[0]
    filePathNameWExt =  path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


def annotateAndSaveAnnotations(r, directory, image_name, file_path, label):
    annotationsJson = annotateResult(r, image_name, file_path, label)
    writeToJSONFile(directory, image_name, annotationsJson)


def annotateImagesInDirectory(rcnn, directory_path, labels):
    for fileName in os.listdir(directory_path):
        if fileName.endswith(".jpg") or fileName.endswith(".jpeg") or fileName.endswith(".png") or fileName.endswith(".tif") or fileName.endswith(".tiff"):
        # load image
            #annotations['filename'] = fileName
            # print("Evaluating Image: " + fileName)
            #file_name_current = fileName
            img = load_img(directory_path+"/"+fileName)
            file_path = directory_path+"/"
            img = img_to_array(img)
            # make prediction
            results = rcnn.detect([img], verbose=0)
            # get dictionary for first prediction
            result = results[0]
            #for l in range(len(labels)):
            # if label is None:
            #     pass
            # else:
            #     labels = [label]

            print('\n--*--\nNow Annotating Image - ' + fileName + '\n')
            print('Annotations: ')

            for l in range(len(labels)):
                if class_names.index(labels[l]) in result['class_ids']:
                    print(str(l+1) + ". " + labels[l] + " found in image! :)")
                    # print("Annotating...")
                    ## Begin Special Code for Kitty ##
                    # if labels[l] == 'cat':
                    #     print('\n\nฅ ̳͒•ˑ̫• ̳͒ฅ\n\nYAY! Kitty found in this one!\n\nฅ ̳͒•ˑ̫• ̳͒ฅ\n\n')
                    ## End Special Code for Kitty ##
                    annotateAndSaveAnnotations(result, directory_path, fileName, file_path, labels[l])
                    if (args.displayMaskedImages is True):
                        display_instances(img, result['rois'], result['masks'], result['class_ids'],
                                    class_names, class_names.index(labels[l]), result['scores'])
                else:
                    print(str(l) + ". " + labels[l] + " not found in image! :(")
                # print("Done Annotating " + '"' + labels[l] + '"' + " in the Image!")
            print("\nFinished Annotating the Image: " + fileName + "\n--*--\n")
            print("\nProcessing...\n")
            time.sleep(3)

ROOT_DIR = os.path.abspath("./")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DATASET_LABELS = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Annotate the object')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'annotateCoco' or 'annotateCustom'")
    parser.add_argument('--image_directory', required=True,
                        metavar="/path/to/the/image/directory/",
                        help='Directory of the images that need to be annotated')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="path_to_weights.h5_file or 'coco_weights'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--label', required=False,
                        metavar="object_label_to_annotate",
                        help='Either COCO dataset labels or custom',
                        default=None)
    parser.add_argument('--labels', required=False,
                        metavar="object_labels_to_annotate",
                        type=bool,
                        help='Allows annotating multiple labels in the image at a time. To use, set --labels=True and add labels to the annotate.py file.', default=False)
    parser.add_argument('--displayMaskedImages', type=bool,
                        default=False, required=False,
                        help='Display the masked images.')

    args = parser.parse_args()

    # Validate arguments
#     if args.command == "annotateCoco":
#       assert args.label in COCO_DATASET_LABELS, "Label --label does not belong to COCO labels "

#   elif args.command == "annotateCustom":
#       assert args.label, "Argument --label is required for annotation"

    assert args.image_directory, "Argument --image_directory is required for annotation"
    assert args.weights, "Argument --weights is required for annotation"

    # labels=[]
    # if args.label is not None:
    if args.labels is True:
        ### SET LABELS HERE ###
        labels = set_labels_here
        # print(labels)
        assert args.label is None, 'Arguments --label and --labels both can not be used at the same time. Please use one.'
        assert labels is not bool(labels), 'No labels are set in annotate.py! Please open the file with any IDE or text editor and add labels in the "set_labels_here" list near the top.'
    elif args.labels is False:
        assert args.label is not None, 'The program can not run without either --label or --labels argument! Please use one.'
        labels = [args.label]
        # print('Arguments --label and --labels both can not be used at the same time. Please use one.')
    # elif args.label is not None:
    elif args.label is not None:
        assert args.labels is False or None, 'Arguments --label and --labels both can not be used at the same time. Please use one.'
        labels = [args.label]
    elif args.labels is True and not labels:
        print('No labels are set in annotate.py! Please open the file with any IDE or text editor and add labels in the line below this one.')
    elif args.label is None and args.labels is False:
        raise Exception("The program can not run without either --label or --labels argument! Please use one.")
    else:
        raise Exception("Please report this exception with the steps to reproduce it!")

    class InferenceCocoConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "inferenceCoco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80

    class InferenceCustomConfig(Config):
        NAME = "inferenceCustom"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1


    if args.command == "annotateCoco":
        config = InferenceCocoConfig()
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    else:
        config = InferenceCustomConfig()
        class_names = ['BG']
        class_names.append(args.label)
    print("\n\n#--------#\n\nProgram Execution Started. All the Best!\n\n#--------#\n\n")
    time.sleep(4)
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config = config, model_dir = "./")

    # Select weights file to load
    if args.command == "annotateCoco":
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights... ", weights_path)
    model.load_weights(weights_path, by_name=True)


    # Annotate
    #file_name_current
    annotations = {
    'filename': '',
    'objects': []
    }
    #annotations = {}


    if args.command == "annotateCoco" or args.command == "annotateCustom":
        #annotateImagesInDirectory(model, directory_path=args.image_directory, label = args.label)
        # labels = args.labels
        # for l in range(len(labels)):
            # annotateImagesInDirectory(model, directory_path=args.image_directory, label = labels[l])
            # annotateImagesInDirectory(model, args.images_directory, labels=[args.label])
        annotateImagesInDirectory(model, args.image_directory, labels)
    else:
        print("'{}' is not recognized. "
              "Use 'annotateCoco' or 'annotateCustom'".format(args.command))
    print("\n\n#--------#\n\nProgram Execution Finished. Congratulations!\n\n#--------#\n\n")

