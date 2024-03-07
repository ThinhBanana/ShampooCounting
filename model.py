import cv2
import os

from object_detect import Detection
from object_classification import Classification


def count_classes(results):
    class_counts = {}
    for result in results:
        class_name = result[4]  # Class name at index 4
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    return class_counts


if __name__ == "__main__":

    # Detection
    weights_path = "./weights/detection.pt"
    image_path = "./test2.jpg"

    # Example of creating the Detection object and calling the detect method
    detector = Detection(weights_path)
    image = cv2.imread(image_path)

    if image is not None:
        img, class_names = detector.detect(image)
    else:
        print(image_path + ' load failed')

    # Get paths of saved cropped images from detection output directory
    detection_output_dir = detector.output_path
    cropped_image_paths = [os.path.join(detection_output_dir, file) for file in os.listdir(detection_output_dir) if
                           file.endswith('.jpg')]

    # Classification
    classifier = Classification()

    # Classify all saved cropped images
    all_results = []
    for cropped_image_path in cropped_image_paths:
        class_name = classifier.classify_image(cv2.imread(cropped_image_path))
        all_results.append((0, 0, 0, 0, class_name))  # Placeholder bounding box coordinates

    # Count occurrences of each class
    class_counts = count_classes(all_results)

    # Display input image with class counts
    for i, (class_name, count) in enumerate(class_counts.items()):
        text = f'{class_name}: {count} objects'
        cv2.putText(img, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
