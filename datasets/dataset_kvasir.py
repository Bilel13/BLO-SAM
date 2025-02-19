import os
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from PIL import Image

class RandomGenerator(object):
    def __init__(self, output_size, low_res, split=None):
        self.output_size = output_size
        self.low_res = low_res
        self.split = split

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        
        # Convert the PIL Image to a NumPy array
        image = np.array(image)
        
        # Resize image if needed
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            scale_factors = (self.output_size[0] / x, self.output_size[1] / y)
            if len(image.shape) == 3:
                scale_factors += (1,)
            image = zoom(image, scale_factors, order=3)

        # Process labels and remove num_labels dimension
        processed_labels = []
        processed_low_res_labels = []
        
        for label_img in labels:
            # Convert PIL label to numpy array
            label = np.array(label_img)
            
            # Resize label to match output size
            label_h, label_w = label.shape
            if label_h != self.output_size[0] or label_w != self.output_size[1]:
                label = zoom(label, (self.output_size[0] / label_h, self.output_size[1] / label_w), order=0)
            
            # Create low-res version
            low_res_label = zoom(label, (self.low_res[0] / self.output_size[0], 
                                       self.low_res[1] / self.output_size[1]), order=0)
            
            processed_labels.append(label)
            processed_low_res_labels.append(low_res_label)

        # Stack labels into single tensors
        stacked_labels = np.stack(processed_labels, axis=0)  # [num_labels, H, W]
        stacked_low_res_labels = np.stack(processed_low_res_labels, axis=0)  # [num_labels, low_res_H, low_res_W]

        # Remove the num_labels dimension
        stacked_labels = stacked_labels[0]  # Select the first label
        stacked_low_res_labels = stacked_low_res_labels[0]  # Select the first low-res label

        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32))
        labels_tensor = torch.from_numpy(stacked_labels.astype(np.float32)).long()
        low_res_labels_tensor = torch.from_numpy(stacked_low_res_labels.astype(np.float32)).long()

        # Handle image dimensions
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)  # Convert to [C, H, W]
        else:
            image = image.unsqueeze(0)  # Convert to [1, H, W]

        sample = {
            'image': image,
            'label': labels_tensor,  # Shape: [H, W]
            'low_res_label': low_res_labels_tensor  # Shape: [low_res_H, low_res_W]
        }
        return sample


class MultiInstanceSegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, dataset=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.dataset = dataset
        self.img_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')

        # Find all corresponding labels for this image
        image_name = self.img_files[idx].split('.')[0]
        label_paths = [f for f in os.listdir(self.label_dir)
                      if f.startswith(image_name) and f.endswith('.jpg')]

        if len(label_paths) == 0:
            raise AssertionError(f"No labels found for {image_name}")

        # Load all labels for this image
        labels = [Image.open(os.path.join(self.label_dir, label_path)).convert('L') 
                 for label_path in sorted(label_paths)]

        sample = {
            'image': img,
            'label': labels
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def collate_fn(batch):
    """
    Collate function to handle batches with stacked labels
    """
    images = []
    labels = []
    low_res_labels = []
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['label'])
        low_res_labels.append(sample['low_res_label'])
    
    # Stack all tensors
    images = torch.stack(images)  # [batch_size, C, H, W]
    labels = torch.stack(labels)  # [batch_size, H, W]
    low_res_labels = torch.stack(low_res_labels)  # [batch_size, low_res_H, low_res_W]
    
    return {
        'image': images,
        'label': labels,  # Shape: [batch_size, H, W]
        'low_res_label': low_res_labels  # Shape: [batch_size, low_res_H, low_res_W]
    }