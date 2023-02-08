# Object Detection Inference with Classification

**Description**

This project is a sample for inference pytorch object detection with tensorflow classification model.

For example, detect human face with object detection model. And then determine the gender of each face from object detection output with classification model.

**Setup**

To setup the model path, input files, and other settings. You should make a json config file like`config.json`.
*  `input_path` specify the input files. It can be a folder or a video.
*  `output_path` specify the image output path. It would be a folder.
*  `model_path` specify the model path.
*  `input_size` specify the input size of the model.
*  `iou_threshold` specify the iou threshold of object detection.
*  `device` specify the device using for the model. It can be `cpu`, `gpu:0`, `gpu:1`, etc. 

**Run**

Specify the config file with `--cfg` and run `inference.py`.

``
python inference.py --cfg=config.json
``
