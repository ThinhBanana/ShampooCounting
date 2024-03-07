import cv2

from ObjectDetection.object_detect_model import DetectYolo
from ObjectClassification.YOLOv5 import Classification

if __name__ == "__main__":

    # DetectYolo
    weights_path = "./weights/best.pt"
    image_path = "./test.jpg"

    # Example of creating the DetectYolo object and calling the detect method
    detector = DetectYolo(weights_path)
    model, device = detector.define()
    image = cv2.imread(image_path)

    if image is not None:
        detector.detect(image, model, device)
    else:
        print(image_path + ' load failed')

    # Classification
    classifier = Classification()
    image_path = "images.jpg"
    results = classifier.classify_image(image_path)
    classifier.display_classification(results, image_path)
