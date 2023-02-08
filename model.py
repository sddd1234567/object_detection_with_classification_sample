# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
import torch
import tensorflow as tf

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, crop_img
from utils.datasets import letterbox

class Model(ABC):
    @abstractmethod
    def preprocessing(self, cfg):
        pass

    @abstractmethod
    def postprocessing(self, cfg):
        pass
    
    @abstractmethod
    def clear(self, cfg):
        pass
    
    @abstractmethod
    def get_device(self, device_name):
        pass


# Pytorch Object Detection Model
class ObjectDetectionModel(Model):
    def __init__(self, cfg):
        self.model_path = cfg['model_path']
        self.input_size = cfg['input_size']
        self.conf_threshold = { name: cfg['objects'][name]['conf_threshold'] for name in cfg['objects']}
        self.iou_threshold = cfg['iou_threshold']
        self.frame_size = None
        self.orig_size = None
        self.device = self.get_device(cfg['device'])
        self.model = attempt_load(self.model_path, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        empty_img = np.transpose(np.zeros(self.input_size), (2, 0, 1))
        empty_img = torch.Tensor(np.expand_dims(empty_img, axis=0)).to(self.device)

        _ = self.model(empty_img) # run once
    
    def __call__(self, image):
        input_tensor = self.preprocessing(image)
        
        # object detection model inference
        detections = self.model(input_tensor, augment = False)[0]
        outputs = self.postprocessing(detections)

        return outputs
    
    def preprocessing(self, image):
        # Preprocess of object detection model
        self.orig_size = image.shape
        image = letterbox(image, new_shape=self.input_size[0])[0] # Padding original img to object detection model input size
        self.frame_size = image.shape
        image = image.transpose(2, 0, 1)
        
        image = np.ascontiguousarray(image)

        input_tensor = torch.from_numpy(image).to(self.device)
        input_tensor = input_tensor.float()
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if input_tensor.ndimension() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        return input_tensor
    
    def postprocessing(self, detections):
        # Postprocess of object detection model
        min_thres = min(self.conf_threshold.values())
        # Apply NMS
        detections = non_max_suppression(detections, 
                                         min_thres, 
                                         self.iou_threshold, 
                                         classes=None,
                                         agnostic=False
                                        )[0]
        
        outputs = {}
        for name in self.names:
            outputs[name] = {
                "boxes": [],
                "scores": [],
                "classes": []
            }
        
        if(detections != None):
            detections[:, :4] = scale_coords(self.frame_size, detections[:, :4], self.orig_size).round() # Scale the prediction bboxe coords into the bbox coords of original input img
            for *xyxy, conf, c in detections:
                class_name = self.names[int(c.cpu())]
                if(conf < self.conf_threshold[class_name]):
                    continue
                
                bbox = [int(xyxy[0].cpu()), int(xyxy[1].cpu()), int(xyxy[2].cpu()), int(xyxy[3].cpu())]
                
                outputs[class_name]["boxes"].append(bbox)
                outputs[class_name]["scores"].append(float(conf.cpu()))
                outputs[class_name]["classes"].append(class_name)
        
        return outputs

    def clear(self):
        del self.model
        
    def get_device(self, device_name):
        return torch.device(device_name.replace("gpu", "cuda"))

# Tensorflow Classification Model
class ClassificationModel(Model):
    def __init__(self, cfg):
        self.input_size = cfg['input_size']
        self.label = cfg['label']
        self.device = self.get_device(cfg['device'])
        
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(cfg['model_path'])

            empty_img = np.expand_dims(np.zeros(self.input_size), axis=0)
            _ = self.model(empty_img) # run once

    def __call__(self, image, bboxes):
        if(len(bboxes) == 0):
            return np.array([])
        
        with tf.device(self.device):
            images = self.preprocessing(image, bboxes)
            preds = self.model.predict(np.array(images))
            preds = self.postprocessing(preds)
        
        return preds
        
    def preprocessing(self, image, bboxes): 
        # Preprocess of classification model
        images = []
        for bbox in bboxes:
            img = crop_img(image, bbox)
            img = tf.image.resize(img, self.input_size[:-1]) 
            img = tf.clip_by_value(img, 0, 255)
            
            images.append(img)
        
        return images
        
    def postprocessing(self, preds):
        # Postprocess of classification model
        pred_label_idxs = np.argmax(preds, axis=1)
        pred_label = [self.label[x] for x in pred_label_idxs] # Multiclass Classification
        
        return pred_label
        
    def clear(self):
        self.model = None
        
    def get_device(self, device_name):
        return device_name