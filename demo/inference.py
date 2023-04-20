import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
TEST_DIR = 'images/'
SAMPLES = 8

PATH_TO_SAVED_MODEL="saved_model/"

print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

print('Done!')


category_index=label_map_util.create_category_index_from_labelmap("/home/label_map.pbtxt",use_display_name=True)

# img=['/home/models/research/test/img (1).jpg']

ext = ['jpg']    # Add image formats here

img = []
[img.extend(glob.glob(TEST_DIR + '*.' + e)) for e in ext]


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

count=0
for image_path in img:
    if(count==SAMPLES):
        break
    if image_path.find(".jpg")!=-1 or  image_path.find(".png")!=-1:
        count= count+1

    print('Running inference for {}... '.format(image_path), end='')
    image_np=load_image_into_numpy_array(image_path)
    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections=detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
                   for key,value in detections.items()}
    detections['num_detections']=num_detections
    detections['detection_classes']=detections['detection_classes'].astype(np.int64)
    image_np_with_detections=image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,     
          min_score_thresh=.7,#0.0001      
          agnostic_mode=False)
    # %matplotlib inline
    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.axis('off')
    plt.show()


    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #save to same folder as data input
    # output_path = f"/home/output/image_{count}.jpg"
    # print(image_path)

    nimg = Image.fromarray(image_np_with_detections)
    nimg.save('/home/output/'+image_path[11:])
