from torch.utils.data import Dataset
import torch
torch.manual_seed(1234)
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from dataset import ImagesDataset


def augment_image(
        img_np: np.ndarray,
        index: int
        ) -> tuple[torch.Tensor, str]:
    transforms_sequence = [
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.RandomRotation((10,180),interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.RandomPerspective(p=1),
                            transforms.RandomSolarize(threshold=0.5, p=1),
                            transforms.RandomInvert(p=1)
                            #transforms.ElasticTransform(100.0)
                        ]
    v = index%(len(transforms_sequence)+3)
    trans_name = None
    img = torch.Tensor(img_np)
    if v == 0:
        trans_name = "Original"
        trans_img = img
    elif v >= 1 and v < 6:
        trans_name = transforms_sequence[v-1].__class__.__name__
        trans_img = transforms_sequence[v-1](img)
    elif v == 6:
        trans_img = F.rotate(transforms.RandomHorizontalFlip(p=1)(img), 90)
        trans_name = "H_Flip+Rotate"
    elif v == 7:
        trans_name = "Compose"
        t = []
        for _ in range(3):
            choice = random.choice(transforms_sequence)
            transforms_sequence.remove(choice)
            t.append(choice)
        t = transforms.Compose(t)
        trans_img = t(img)
    return trans_img, trans_name

class TransformedImagesDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.num_of_transforms = 8 #1 original, 5 simple, 1 custom, 1 compose

    def __getitem__(self, index: int):
        i = index//self.num_of_transforms
        j = index%self.num_of_transforms
        trans_img, trans_name = augment_image(self.dataset[i][0], j)
        return trans_img, self.dataset[i][1], trans_name, index, self.dataset[i][2], self.dataset[i][3]

    def __len__(self):
        return len(self.dataset)*self.num_of_transforms

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dataset = ImagesDataset("./training_data", 100, 100, float)
    transformed_ds = TransformedImagesDataset(dataset)
    fig, axes = plt.subplots(2, 4)
    for i in range(0, 8):
        trans_img, classid, trans_name, index, classname, img_path = transformed_ds.__getitem__(i)
        _i = i // 4
        _j = i % 4
        axes[_i, _j].imshow(transforms.functional.to_pil_image(trans_img),cmap="gray_r")
        axes[_i, _j].set_title(f'{trans_name}\n{classname}')
    fig.tight_layout()
    plt.savefig("images")
