import torch, torchvision, utils, argparse, os
from criteria import *
from utils import upscale, get_resnet18
from generator import FedLeak
import pandas as pd
from PIL import Image


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, dir="./imagenet/images", csv_path="./imagenet/images.csv", transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, Truelabel

    def __len__(self):
        return len(self.csv)

class NormalizedModel(torch.nn.Module):
    def __init__(self, model, mean, std, device):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.resize_op = torchvision.transforms.Resize((224, 224))
        self.mean, self.std = mean.to(device), std.to(device)

    def forward(self, x):
        x = self.resize_op(x)
        x = (x - self.mean) / self.std
        return self.model(x)


def get_client_gradient(data, label, model, loss_fn):
    loss = loss_fn(model(data), label)
    grads = torch.autograd.grad(loss, model.parameters())
    return grads


def l1_loss(x, y):
    return (x-y).abs().mean()


def l2_loss(x, y):
    return (x-y).norm(2)


def cos_loss(x, y):
    mask = x.abs().detach() >= x.detach().flatten().abs().quantile(0.2).item() # 只保留最小的百分之多少
    x, y = x * mask, y * mask
    return -(x.flatten() * y.flatten()).sum() / (x.flatten().norm(2)+1e-8) / (y.flatten().norm(2)+1e-8)


def combine_loss(x, y):
    mask = (y.abs() >= torch.quantile(y.abs(), 0.5).item()).float()
    x, y = mask * x, mask * y
    return l1_loss(x, y) + cos_loss(x, y)


if __name__ == "__main__":

    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:0")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize((128, 128))])
    trainloader = torch.utils.data.DataLoader(ImageNet(transforms=transform), batch_size=16, shuffle=True)
    model = get_resnet18().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    data, label = next(iter(trainloader))
    data, label = data.to(device), label.to(device)
    torchvision.utils.save_image(data, f"./real_imgs.png")
    client_grads = get_client_gradient(data, label, model, torch.nn.CrossEntropyLoss())
    client_grads = [grad.clone() for grad in client_grads]

    data = upscale(data)
    best_ones_PSNR = torch.zeros_like(data)
    best_ones_SSIM = torch.zeros_like(data)
    best_ones_LPIPS = torch.zeros_like(data)

    for _ in range(1): # restart for better attack performance.
        recovered_images = FedLeak(client_grads, label, model, combine_loss)
        best_ones_PSNR = update_best_ones(best_ones_PSNR, data, recovered_images, "PSNR")

    torchvision.utils.save_image(best_ones_PSNR, f"./recovered_imgs.png")
    print(compute_psnr(data, best_ones_PSNR, False).mean().item())