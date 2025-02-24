import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as t
from utils import tokenizer  # assuming your tokenizer function remains unchanged


class Animal5(Dataset):
    def __init__(self, root_dir, train=True, caption_map=None):
        """
        Args:
            root_dir (str): Path to the root directory of your dataset.
            train (bool): Whether to load the training set (True) or test set (False).
            caption_map (dict, optional): A dictionary mapping category names to custom captions.
                                           If not provided, a default caption is generated.
        """
        self.split = "train" if train else "test"
        self.data_dir = os.path.join(root_dir, self.split)
        self.transform = t.ToTensor()

        # Load all image paths and their corresponding category names.
        self.data = []
        for category in os.listdir(self.data_dir):
            category_path = os.path.join(self.data_dir, category)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(category_path, img_file)
                        self.data.append((image_path, category))

        # Store the provided caption mapping or use an empty dictionary.
        self.caption_map = caption_map if caption_map is not None else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, category = self.data[idx]

        # Open the image and ensure it's in RGB format.
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        # Use custom caption if available, otherwise default caption.
        caption_text = self.caption_map.get(category, f"An image of {category}")
        cap, mask = tokenizer(caption_text)
        mask = mask.repeat(len(mask), 1)

        return {"image": img, "caption": cap, "mask": mask}
