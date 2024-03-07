import glob
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


class DetectYolo:
    def __init__(self, weights, conf_thres=0.25, iou_thres=0.45, x_offset=0.2, y_offset=0.15, crop_save=True,
                 output_path='./output'):
        self.weights_path = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.crop_save = crop_save
        self.output_path = output_path

    def draw_line(self, image, xf1, yf1, xf2, yf2):
        w = image.shape[1]
        h = image.shape[0]

        start_point = (int(w * xf1), int(h * yf1))
        end_point = (int(w * xf2), int(h * yf2))

        cv2.line(image, start_point, end_point, (255, 0, 0), 7)

        return xf1, yf1

    def is_inside(self, polygon, centroid):
        centroid = Point(centroid)
        return polygon.contains(centroid)

    def draw_bounding_boxes(self, image, file_path, boxes, polygon):
        w = image.shape[1]
        h = image.shape[0]

        class_names = ['object']  # Replace with your own class names

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            if self.is_inside(polygon, (x1 / w, y1 / h)) or self.is_inside(polygon, (x2 / w, y2 / h)):
                self.save_one_box(boxes[i], image,)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return image

    def xyxy2xywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def clip_coords(self, boxes, img_shape, step=2):
        boxes[:, 0::step].clamp_(0, img_shape[1])
        boxes[:, 1::step].clamp_(0, img_shape[0])

    def save_one_box(self, xyxy, im, output_path='output/', file='output.jpg', gain=1, pad=2, square=False, BGR=False):
        xyxy = torch.tensor(xyxy).view(-1, 4)
        b = self.xyxy2xywh(xyxy)
        if square:
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
        b[:, 2:] = b[:, 2:] * gain + pad
        xyxy = self.xywh2xyxy(b).long()
        self.clip_coords(xyxy, im.shape)
        crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]

        if self.crop_save:
            output_file = str(self.increment_path(output_path + file, mkdir=True).with_suffix('.jpg'))
            cv2.imwrite(output_file, crop if BGR else crop[..., ::-1])

        return crop

    def increment_path(self, path, exist_ok=False, sep='', mkdir=False):
        path = Path(path)
        if path.exists() and not exist_ok:
            suffix = path.suffix
            path = path.with_suffix('')
            dirs = glob.glob(f"{path}{sep}*")
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            path = Path(f"{path}{sep}{n}{suffix}")
        dir = path if path.suffix == '' else path.parent
        if not dir.exists() and mkdir:
            dir.mkdir(parents=True, exist_ok=True)
        return path

    def define(self):
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(self.weights_path, device=device)
        return model, device

    def detect(self, detect_image, model, device):
        # Draw line and add points of the area
        # New line start point must be the end of the old line
        points = [self.draw_line(detect_image, self.x_offset, self.y_offset, 1 - self.x_offset, self.y_offset),
                  self.draw_line(detect_image, 1 - self.x_offset, self.y_offset, 1 - self.x_offset, 1 - self.y_offset),
                  self.draw_line(detect_image, 1 - self.x_offset, 1 - self.y_offset, self.x_offset, 1 - self.y_offset),
                  self.draw_line(detect_image, self.x_offset, 1 - self.y_offset, self.x_offset, self.y_offset)]

        points = list(dict.fromkeys(points))

        # Preprocess the image
        # Convert the image from BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image_rgb)

        shape = image.size
        image = image.resize((640, 640))
        img = np.array(image)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        # Run the YOLOv5 model on the image
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # convert to numpy
        pred = [x.detach().cpu().numpy() for x in pred]
        # convert to int
        pred = [x.astype(int) for x in pred]
        # Post-process the output and draw bounding boxes on the image
        boxes = []
        confidences = []
        class_ids = []
        for det in pred:
            if det is not None and len(det):
                # Scale the bounding box coordinates to the original image size
                det[:, :4] = det[:, :4] / 640 * image.size[0]
                for *xyxy, conf, cls in det:
                    boxes.append(xyxy)
                    confidences.append(conf.item())
                    class_ids.append(int(cls.item()))
        image = np.array(image)

        image = self.draw_bounding_boxes(image, self.output_path, boxes, Polygon(points))

        # Blue out-side offset area
        blurred_img = cv2.GaussianBlur(image, (21, 21), 0)

        mask = np.zeros((640, 640, 3), dtype=np.uint8)

        # For latest OpenCV
        mask = cv2.rectangle(mask, (int(640 * self.x_offset), int(640 * self.y_offset)),
                             (int(640 * (1 - self.x_offset)), int(640 * (1 - self.y_offset))), (255, 255, 255), -1)
        image = np.where(mask == (255, 255, 255), image, blurred_img)

        image = Image.fromarray(image)
        # resize back to original size
        image = image.resize(shape)
        # clear memory and cache
        torch.cuda.empty_cache()

        print("The image was successfully cropped")

    def scale_coords(self, img1_shape, coords, img0_shape):
        gain = max(img1_shape) / max(img0_shape)
        pad = (max(img1_shape) - img0_shape[1], max(img1_shape) - img0_shape[0])
        coords[:, [0, 2]] -= pad[0] / 2
        coords[:, [1, 3]] -= pad[1] / 2
        coords[:, :4] /= gain
        self.clip_coords(coords, img1_shape)
        return coords


# Example usage
if __name__ == "__main__":
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
