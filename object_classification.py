from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import ImageDraw, ImageFont
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225

model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="./weights/classification.pt",
                       force_reload=True)


class Classification:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="./weights/classification.pt",
                                    force_reload=True)
        self.model.eval()

    def classify_transforms(self, size=224):
        return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def classify_image(self, image):
        image_pil = Image.fromarray(image)  # Convert NumPy array to PIL Image
        transformations = self.classify_transforms()
        convert_tensor = transformations(image_pil)
        convert_tensor = convert_tensor.unsqueeze(0)
        convert_tensor = convert_tensor.to(device)

        results = self.model(convert_tensor)
        pred = F.softmax(results, dim=1)
        top_class = pred.argmax(dim=1).item()  # Get the index of the class with the highest probability

        class_name = self.model.names[top_class]

        return str(class_name)  # Ensure class name is returned as string

    def display_classification(self, results, image_path):
        pred = F.softmax(results, dim=1)
        for i, prob in enumerate(pred):
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            text = '\n'.join(f'{prob[j]:.2f} {self.model.names[j]}' for j in top5i)
            print(text)

        img = Image.open(image_path)
        I1 = ImageDraw.Draw(img)
        myFont = ImageFont.truetype('arial.ttf', 75)
        I1.text((10, 10), text, font=myFont, fill=(255, 255, 255))
        img.show()


if __name__ == "__main__":
    classifier = Classification()
    image_path = "images.jpg"
    results = classifier.classify_image(image_path)
    classifier.display_classification(results, image_path)
