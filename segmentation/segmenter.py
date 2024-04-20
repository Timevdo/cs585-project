from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms


class SegmentedImage:
    segmented_image: np.ndarray
    class_list: list

    def __init__(self, predicted_img, class_dict_path):
        self.segmented_image = predicted_img
        parsed_dict = self.__parse_class_dict(class_dict_path)
        self.class_list = [parsed_dict[i] for i in range(len(parsed_dict))]

    def get_class_name(self, x, y):
        class_id = self.segmented_image[y, x]
        return self.class_list[class_id][1]

    def get_class_color(self, x, y):
        class_id = self.segmented_image[y, x]
        return self.class_list[class_id][0]

    def get_rgb_image(self):
        pred_img = np.zeros((self.segmented_image.shape[0], self.segmented_image.shape[1], 3), dtype=np.uint8)
        for i in range(self.segmented_image.shape[0]):
            for j in range(self.segmented_image.shape[1]):
                pred_img[i][j] = self.__id_to_rgb(self.segmented_image[i][j])
        return pred_img

    @staticmethod
    def __parse_class_dict(class_dict_path):
        id_dict = {}
        df = pd.read_csv(class_dict_path)
        for i in range(len(df)):
            class_id = i
            color = (df.iloc[i]['r'], df.iloc[i]['g'], df.iloc[i]['b'])
            class_name = df.iloc[i]['name']
            id_dict[class_id] = (color, class_name)
        return id_dict

    def __id_to_rgb(self, id):
        return self.class_list[id][0]

    def __id_to_name(self, id):
        return self.class_list[id][1]


class Segmenter:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model: Any
    device: Any
    class_dict_path: str

    def __init__(self, model_path: str, class_dict_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.model.eval()
        self.class_dict_path = class_dict_path

    def segment_image(self, image_frame):
        # Convert the image to rgb from bgr
        image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        # Convert the image to a tensor
        image_tensor = transforms.ToTensor()(image_frame)
        # Normalize the image
        image_tensor = self.normalize(image_tensor)
        # Pass the image through the model
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        # Convert the predicted image to a numpy array
        pred = predicted[0].cpu().numpy()
        return SegmentedImage(pred, self.class_dict_path)


if __name__ == "__main__":
    # Read image
    frame = cv2.imread("./Seq05VD_f00510.png")
    segmenter = Segmenter("segmentation-model.pt", "class_dict.csv")
    segmented_image = segmenter.segment_image(frame)

    cv2.imshow("Segmented Image", segmented_image.get_rgb_image())
    cv2.waitKey(0)
