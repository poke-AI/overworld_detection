import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
import cv2

from torchvision.datasets import VisionDataset


class OverworldDataset(VisionDataset):
    def __init__(self, root, transforms=None):
        super(OverworldDataset, self).__init__(root, transforms)
        self.root = root
        self.transforms = transforms
        temp_xml_file_names = glob.glob(os.path.join(os.path.abspath(root), "*.xml"))

        self.xml_file_names = []
        self.jpg_file_names = []

        for xml_file in temp_xml_file_names:
            xml = ET.parse(xml_file).getroot()
            if len(xml.findall("object")) <= 0:
                continue
            
            dir = os.path.dirname(xml_file)
            base_name = os.path.basename(xml_file).split(".")[0] + ".jpg"
            self.jpg_file_names.append(os.path.join(dir, base_name))
            self.xml_file_names.append(xml_file)

        self.label_dict = {
            "pokecen": 0,
            "pokemart": 1,
            "npc": 2,
            "house": 3,
            "gym": 4,
            "exit": 5,
            "wall": 6,
            "grass": 7
        }


    def __getitem__(self, index):
        img = cv2.imread(self.jpg_file_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0

        img = np.moveaxis(img, 2, 0)
        image = torch.Tensor(img)

        xml = ET.parse(self.xml_file_names[index]).getroot()

        target = {  
            "image_id": self.jpg_file_names[index],
            "labels": [],
            "boxes": []
        }
        for member in xml.findall("object"):
            target["labels"].append(self.label_dict[member[0].text])
            target["boxes"].append([
                int(member[4][0].text) / 720.0,
                int(member[4][1].text) / 720.0,
                int(member[4][2].text) / 720.0,
                int(member[4][3].text) / 720.0
            ])
        
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        return image, target

    
    def __len__(self):
        assert len(self.jpg_file_names) == len(self.xml_file_names)
        return len(self.jpg_file_names)
    

if __name__ == "__main__":
    dataset = OverworldDataset("train")