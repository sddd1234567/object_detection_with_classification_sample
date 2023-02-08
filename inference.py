import os
import cv2
import glob
import json
import argparse

from config import cfg
from utils.general import plot_one_box
from model import ObjectDetectionModel, ClassificationModel

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

def load_config(cfg_file):
    with open(cfg_file) as json_file:
        data = json.load(json_file)
    
    return data
        
def plot_bboxes(img, det, label_colors):
    for i in range(len(det['boxes'])):
        if(det['classes'][i] in label_colors):
            plot_one_box(det['boxes'][i], img, label=det['classes'][i], color=label_colors[det['classes'][i]], line_thickness=4)
    
    return img


def draw_result(img, preds, label_colors):
    for label_name in preds:            
        img = plot_bboxes(img, preds[label_name], label_colors=label_colors)

    return img


def inference(img, object_detection_model, classification_models):
    # Apply object detection
    preds = object_detection_model(img)

    # Apply classification after object detection
    for label in classification_models:
        # Apply classification to the predicted bboxes from object detection.
        class_result = classification_models[label](img, preds[label]['boxes'])
        preds[label]['classes'] = [pred_label for pred_label in class_result] # update class label

    return preds

def detect_video(object_detection_model, classification_models, video_path, output_folder, label_colors):
    cap = cv2.VideoCapture(video_path) # Load video source
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(output_folder, os.path.basename(opt.source))
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))        
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
    while(cap.isOpened()):
        ret, input_img = cap.read()
        if(not ret):
            break
            
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        preds = inference(input_img, object_detection_model, classification_models) # Inference object detection and classification
        output_img = draw_result(input_img, preds, label_colors) # Draw detection result to img

        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR) # Convert into bgr for cv2 save
        vid_writer.write(output_img) # Save video

    cap.release()
    vid_writer.release()
    

def detect_img_folder(object_detection_model, classification_models, folder_path, output_folder, label_colors):
    images = list(glob.iglob(os.path.join(folder_path, "*")))
    images = [x for x in images if os.path.splitext(x)[-1].lower() in img_formats]
    
    for img_path in images:
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        
        input_img = cv2.imread(img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        preds = inference(input_img, object_detection_model, classification_models) # Inference object detection and classification
        output_img = draw_result(input_img, preds, label_colors) # Draw detection result to img

        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR) # Convert into bgr for cv2 save
        cv2.imwrite(output_path, output_img) # Save image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config.json', help='json config file')
    opt = parser.parse_args()
    
    cfg = load_config(opt.cfg)

    object_detection_model = ObjectDetectionModel(cfg)
    classification_models = {label: ClassificationModel(cfg=cfg['objects'][label]['classification']) for label in cfg['objects'] if (not cfg['objects'][label]['classification'] is None)}
    
    # Check if the input_path is a video or a folder
    is_video = (os.path.splitext(cfg['input_path'])[-1].lower() in vid_formats)
    
    if(is_video):
        detect_video(object_detection_model, classification_models, cfg['input_path'], cfg['output_path'], cfg['label_colors'])
    else:
        detect_img_folder(object_detection_model, classification_models, cfg['input_path'], cfg['output_path'], cfg['label_colors'])
            
    print("Done")