import pathlib

# Temporarily modify the PosixPath for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class Classification:
    def __init__(self, model_path="best.pt"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        self.names = self.model.names

    def classify_transforms(self, size=224):
        return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])

    def classify_image(self, image_path):
        image = Image.open(image_path)
        transformations = self.classify_transforms()
        convert_tensor = transformations(image)
        convert_tensor = convert_tensor.unsqueeze(0)
        convert_tensor = convert_tensor.to(self.device)
        results = self.model(convert_tensor)
        return results

    def display_classification(self, results, image_path):
        pred = torch.nn.functional.softmax(results, dim=1)
        img = Image.open(image_path)
        I1 = ImageDraw.Draw(img)
        myFont = ImageFont.truetype('arial.ttf', 75)
        for i, prob in enumerate(pred):
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            text = '\n'.join(f'{prob[j]:.2f} {self.names[j]}' for j in top5i)
            I1.text((10, 10), text, font=myFont, fill=(255, 255, 255))
        img.show()

if __name__ == "__main__":
    classifier = Classification()
    image_path = "images.jpg"
    results = classifier.classify_image(image_path)
    classifier.display_classification(results, image_path)
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import pathlib

# Temporarily modify the PosixPath for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class Classification:
    def __init__(self, model_path="best.pt"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        self.names = self.model.names

    def classify_transforms(self, size=224):
        return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])

    def classify_image(self, image_path):
        image = Image.open(image_path)
        transformations = self.classify_transforms()
        convert_tensor = transformations(image)
        convert_tensor = convert_tensor.unsqueeze(0)
        convert_tensor = convert_tensor.to(self.device)
        results = self.model(convert_tensor)
        return results

    def display_classification(self, results, image_path):
        pred = torch.nn.functional.softmax(results, dim=1)
        img = Image.open(image_path)
        I1 = ImageDraw.Draw(img)
        myFont = ImageFont.truetype('arial.ttf', 75)
        for i, prob in enumerate(pred):
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            text = '\n'.join(f'{prob[j]:.2f} {self.names[j]}' for j in top5i)
            I1.text((10, 10), text, font=myFont, fill=(255, 255, 255))
        img.show()

if __name__ == "__main__":
    classifier = Classification()
    image_path = "images.jpg"
    results = classifier.classify_image(image_path)
    classifier.display_classification(results, image_path)
