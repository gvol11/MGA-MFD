#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 06:22:40 2021

@author: carmis
"""

import matplotlib.pyplot as plt
import matplotlib
# import required functions, classes
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image, visualize_object_predictions, ipython_display
from sahi.predict import get_prediction, get_sliced_prediction, predict

detection_model = Yolov5DetectionModel(
    model_path='/data2/mritime/yolov5/yolov5/runs/train/6epochs_1024_smd_hous1_charls0/weights/best.pt',
    prediction_score_threshold=0.3,
    device="cuda", # or 'cpu'
)

image_dir = "/data2/mritime/youtube_videos/houston_1080/fr_136290.png"
image = read_image(image_dir)

# result = get_prediction(image, detection_model)


def vis_pred():
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    visualization_result = visualize_object_predictions(
        image,
        object_prediction_list=result["object_prediction_list"],
        output_dir=None,
        file_name=None,
    )
    matplotlib.use('TkAgg')
    plt.imshow(visualization_result["image"])
    plt.show()


def batch_pred():
    model_name = "Yolov5DetectionModel"
    model_parameters = {
        "model_path": '/data2/mritime/yolov5/yolov5/runs/train/6epochs_1024_smd_hous1_charls0/weights/best.pt',
        "device": "cuda",
        "prediction_score_threshold": 0.4,

    }
    apply_sliced_prediction = True
    slice_height = 1024
    slice_width = 1024
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2
    export_crop = True
    source_image_dir = "/data2/mritime/youtube_videos/houston_1080"

    predict(
        model_name=model_name,
        model_parameters=model_parameters,
        source=source_image_dir,
        export_crop=export_crop,
        apply_sliced_prediction=apply_sliced_prediction,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )


if __name__ == '__main__':
    batch_pred()
    # vis_pred()