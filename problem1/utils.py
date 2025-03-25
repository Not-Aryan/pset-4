import matplotlib.pyplot as plt
import torch
from PIL import Image
from skimage.data import astronaut
from skimage.color import rgb2gray
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import random
import numpy as np

def set_seed(seed: int = 42):
    """Enable reproducibility by setting the seed for random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    def __init__(self, height):
        super().__init__()
        self.height = height
        self.img = self.get_image(height)
        self.pixels = self.img.permute(1, 2, 0).view(-1, 1)
        self.coords = self.get_mgrid(height, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError
        return self.coords, self.pixels

    def get_image(self, height):
        img = Image.fromarray(rgb2gray(astronaut()))
        transform = Compose(
            [
                Resize(height),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )
        img = transform(img)
        return img

    def get_mgrid(self, height, dim=2):
        """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=height)])
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid
    

def psnr(model_output: torch.Tensor, ground_truth: torch.Tensor):
    mse = torch.nn.functional.mse_loss(model_output, ground_truth)
    return 10 * torch.log10(1 / mse)


def plot(dataset: ImageDataset, model_output: torch.Tensor, img_grad: torch.Tensor, img_laplacian: torch.Tensor):
    img = dataset.img.cpu().view(dataset.height, dataset.height)

    _, axs = plt.subplots(ncols=4, figsize=(18, 6))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(model_output.cpu().detach().view(dataset.height, dataset.height), cmap="gray")
    axs[1].set_title("Model Output")
    axs[2].imshow(img_grad.norm(dim=-1).cpu().detach().view(dataset.height, dataset.height), cmap="turbo")
    axs[2].set_title("Image Gradient")
    axs[3].imshow(img_laplacian.cpu().detach().view(dataset.height, dataset.height), cmap="turbo")
    axs[3].set_title("Image Laplacian")
    plt.tight_layout()
    plt.show()