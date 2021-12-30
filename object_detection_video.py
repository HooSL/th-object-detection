import tensorflow as tf
import os
import pathlib

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from plot_object_detection_saved_model import IMAGE_PATHS

import time


#내 로컬에 설치된 레이블 파일을, 인덱스와 연결시킨다
PATH_TO_LABELS = 'C:\\Users\\5-1\\Documents\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)


#모델 로드하는 함수
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
#위 링크에서 모델을 가져올 수 있다

#테스트해본 모델들 (링크주소 복사하기)
#/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
#/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
#/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
#/20200711/efficientdet_d0_coco17_tpu-32.tar.gz

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/' #경로는 변하지 않음
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)
#모델 날짜와 모델 이름만 바뀜
MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)



def load_model(model_dir):
    model_full_dir = model_dir + "/saved_model"

    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)


#동영상 출력하는 함수
def show_inference(detection_model,image_np) :

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detection_model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    print(detections)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow('result',image_np_with_detections)


#동영상 저장하는 함수
def save_inference(detection_model,image_np,video_writer):
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detection_model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    print(detections)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    video_writer.write(image_np_with_detections)



#비디오를 실행하는 코드
cap = cv2.VideoCapture('data/video.mp4')

#비디오 촬영하는 코드
#cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('비디오 실행 애러')
else:

    #동영상 저장하는 변수 코드
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('data/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                            20,(frame_width, frame_height))


    #비디오 캡쳐에서 이미지를 1장씩 가져온다
    #이 1장의 이미지를 오브젝트 디텍션한다
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == True:
            #frame이 이미지에 대한 넘파이 어레이 이므로 이 frame을 오브젝트 디텍션한다

            #불러오는 시간 체크
            start_time = time.time()

            #동영상을 디텍션 한 후 파일로 저장하는 함수
            save_inference(detection_model,frame,out)

            #동영상을 바로 실시간으로 화면에 디텍팅하는 코드
            #show_inference(detection_model,frame)

            end_time = time.time()

            print('연산에 걸린 시간',str(end_time-start_time))

            if cv2.waitKey(27) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyWindow()


























