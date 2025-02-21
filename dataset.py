from torch.utils.data import Dataset
from datasets import load_dataset
from utils import tokenizer
import torchvision.transforms as t

class FashionMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("fashion_mnist")

        self.transform = t.ToTensor()

        if train:
            self.split = "train"
        else:
            self.split = "test"


        self.captions = {0: "An image of a t-shirt/top",
                        1: "An image of trousers",
                        2: "An image of a pullover",
                        3: "An image of a dress",
                        4: "An image of a coat",
                        5: "An image of a sandal",
                        6: "An image of a shirt",
                        7: "An image of a sneaker",
                        8: "An image of a bag",
                        9: "An image of an ankle boot"}


    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self,i):
        img = self.dataset[self.split][i]["image"]
        img = self.transform(img)

        cap, mask = tokenizer(self.captions[self.dataset[self.split][i]["label"]])

        mask = mask.repeat(len(mask),1)

        return {"image": img, "caption": cap, "mask": mask}