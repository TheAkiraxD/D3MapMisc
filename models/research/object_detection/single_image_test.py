import numpy as np
import os
import sys
import tensorflow as tf
import time
import math
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util

MODEL_NAME = 'map_misc_finder2'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 7) ]

t_detection_boxes = None
t_detection_masks = None
t_real_num_detection = None
last_time = time.time()

def truncate(f, n):
  return math.floor(f * 10 ** n) / 10 ** n
  
def run_inference_for_single_image(image, graph, sess):
  with graph.as_default():
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
      output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
    
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def find_object(image_x, image_y, boxes, classes, scores, category_index, max_boxes_to_draw = 5, min_score_thresh = .6):
  objects = []  # x1, y1, x2, y2, index, %
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is not None and scores[i] > min_score_thresh:
      item= []
      item.append(int(boxes[i][1]*image_x))
      item.append(int(boxes[i][0]*image_y))
      item.append(int(boxes[i][3]*image_x))
      item.append(int(boxes[i][2]*image_y))
      
      if classes[i] in category_index.keys():
        item.append(classes[i])
        item.append(int(truncate(scores[i], 2) * 100))
      objects.append(item)
  return objects

        
with tf.Session(graph=detection_graph) as sess:
  for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph, sess)
    dim = image_np[0][0].size
    image_size_x = int(image_np[0].size / dim)
    image_size_y = int(image_np.size / (image_size_x*dim))
    
    print(str(image_path[-10:]) + ' - ' + str(find_object(image_size_x, image_size_y, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], category_index)))
    print('Loop took {} seconds'.format(truncate((time.time()-last_time),3)))
    last_time = time.time()



