import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from torch.utils.data import Dataset


# use https://myvision.ai/ to create xml files for images

# the dataset class
class TreeDataset(Dataset):

    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        num_objs = len(root.findall('object'))


        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros([0], dtype=torch.int64)
            area = torch.zeros([0], dtype=torch.float32)
            iscrowd = torch.zeros([0], dtype=torch.int64)

        
        else:
            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall('object'):
                # map the current object name to `classes` list to get...
                # ... the label index and append to `labels` list
                labels.append(self.classes.index(member.find('name').text))
                
                # xmin = left corner x-coordinates
                xmin = int(member.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                xmax = int(member.find('bndbox').find('xmax').text)
                # ymin = left corner y-coordinates
                ymin = int(member.find('bndbox').find('ymin').text)
                # ymax = right corner y-coordinates
                ymax = int(member.find('bndbox').find('ymax').text)
                
                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = (xmin/image_width)*self.width
                xmax_final = (xmax/image_width)*self.width
                ymin_final = (ymin/image_height)*self.height
                yamx_final = (ymax/image_height)*self.height
                
                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # labels to tensor
            #labels = torch.ones((num_objs,), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            
        return image_resized, target

    
    def __len__(self):
        return len(self.all_images)
    