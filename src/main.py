import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np

def load_image(image_path):
    return Image.open(image_path)

def preprocess_image(image):
    transform = T.ToTensor()
    return transform(image)

def detect_objects(model, image_tensor):
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction[0]

def draw_boxes(image_path, bboxes, labels, scores, threshold=0.8):
    image = cv2.imread(image_path)
    coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                  "train", "truck", "boat", "traffic light", "fire hydrant",
                  "street sign", "stop sign", "parking meter", "bench", "bird",
                  "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                  "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                  "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
                  "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket",
                  "bottle", "plate", "wine glass", "cup", "fork", "knife",
                  "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                  "chair", "couch", "potted plant", "bed", "mirror", "dining table",
                  "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                  "sink", "refrigerator", "blender", "book", "clock", "vase",
                  "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

    for i in range(len(scores)):
        if scores[i] > threshold:
            x1, y1, x2, y2 = bboxes[i].numpy().astype(int)
            label = coco_names[labels[i] - 1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image_path = "data/traffic.jpg"
    image = load_image(image_path)
    image_tensor = preprocess_image(image)
    prediction = detect_objects(model, image_tensor)
    bboxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
    draw_boxes(image_path, bboxes, labels, scores)

if __name__ == "__main__":
    main()