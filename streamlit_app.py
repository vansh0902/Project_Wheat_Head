
import streamlit as st
import pandas as pd
# import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
# import cv2
import numpy as np
import time
import sys

import tensorflow as tf
import time
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import visualization_utils as viz_utils

def main():
    new_title = '<p style="font-size: 42px;">Welcome to my Object Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate Object detection in images.
    DL model used for this project -  SSD_Mobilenet
    
    
    This object detection algorithm counts the number of wheat heads present in an image"""
    )
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)"))
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    
    if choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        object_detection_image()

def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image
    """)
#     file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    
    with st.form("my_form"):
	file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
	   
	submitted = st.form_submit_button("Predict")
	if submitted:
	    st.write("Execution begin")


    if file!= None:
        image_path = Image.open(file)
        img2 = np.array(image_path)

        st.image(image_path, caption = "Uploaded Image")
        my_bar = st.progress(0)
        nmsThreshold= st.slider('Threshold', 0, 100, 40)
        classNames = ["Wheat Head"]
        whT = 320
    else:
	st.write(

    def load_image_into_numpy_array(image):
        return np.array(image)

    IMAGE_SIZE = (12, 8) # Output display size as you want
    PATH_TO_SAVED_MODEL=r'saved_model_mobile/saved_model'

    # Load saved model and build the detection function
    detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

    #Loading the label_map
    category_index= {1: {'id': 1, 'name': 'WheatHead'}}
    image_np = load_image_into_numpy_array(image_path)



    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()          

    box_to_color_map = viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=nmsThreshold/100, # Adjust this value to set the minimum probability boxes to be classified as True
      agnostic_mode=False)
        
    # findObjects(detections,image_np_with_detections)
    number = len(box_to_color_map)

    st.image(image_np_with_detections, caption='Proccesed Image.')
    st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {number}')
    
    my_bar.progress(100)


if __name__ == '__main__':
		main()	
