import streamlit as st

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import time

import tensorflow as tf
import pathlib
import cv2
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

def load_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_date + '/' + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict


def show_inference(model, image_path, isImage=True):
    st.spinner()
    with st.spinner(text='모델 로드 완료! 분석을 시작합니다.'):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #print(image_path)
        #return
        startTime = time.time()
        print(startTime)
        # 영상 캡쳐 일때 처리 (이미 cv2로 열어서 오기 때문에 따로 실행 안함)
        if isImage == False:
            image_np = image_path   #이미 동영상 처리에서 cap으로 열린 상태
        else :  # True 이미지일 경우
            image_np = np.array(Image.open(image_path)) # Image.open()은 RGB로 읽어들임. cv2라이브러리만 빼고 (BGR)
        
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.array(output_dict['detection_boxes']),
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed',None),
            use_normalized_coordinates=True,
            line_thickness=8)
        endTime = time.time()
        print(endTime-startTime)
        st.success('분석이 완료 되었습니다.')
        #함수 호출한 곳에서 cv2.write()을 사용하기 때문에 (처음에 cv2로 읽음 그대로 리턴해주면 색이 제대로 저장 나옴)
        image_np_only_show = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  #st.image로 보여줄려면은 주석해제, BGR에서 RGB로 바꿔야함
        #st.image로 보여주고 후 리턴은 다시 BGR로 바꿔줘야함
        st.image(image_np_only_show)
        # 리턴은 원래 BGR형태로 // reCaptureVideoTfod()함수에서 cv에서 다시 저장함
        return image_np