# Faster R-CNN Implementation
This repository contains an implementation of Faster R-CNN for object detection using a pre-trained `fasterrcnn_resnet50_fpn` model from `torchvision`. The model is capable of detecting objects in an image and drawing bounding boxes around them.

## Flow:
- Uses a pre-trained Faster R-CNN model.
- Detects objects in an image.
- Draws bounding boxes with labels on detected objects.
- Classifies the objects into a class.

## Requirements
- Python 3.x
- torch
- torchvision
- numpy
- cv2

## Usage
1. Clone the repository:
```
git clone https://github.com/knowmili/faster-rcnn.git
cd faster-rcnn
```

## Output
The script will display the image with bounding boxes and labels drawn around detected objects.