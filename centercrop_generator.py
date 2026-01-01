import time
from typing import Callable

import torch, torchvision, utils
import torch.nn as nn
import torchvision.transforms.functional as TF
import layers
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import upscale


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super().__init__()
        convs = [layers.SNConv2d(in_channel, out_channel, kernel_size, padding=padding)]
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        convs.append(layers.SNConv2d(out_channel, out_channel, kernel_size, padding=padding))
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def label_to_onehot(target, num_classes=1000):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def get_indices(grads, indice_num):
    _, indices = torch.topk(torch.stack([p.norm() for p in grads], dim=0), indice_num) # 总共是34
    return indices


class Generator(nn.Module):    
    def __init__(self, image_res=32, input_code_dim=128, in_channel=256, tanh=True):
        super().__init__()
        self.image_res = image_res
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1)  # , pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1)  # , pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1)  # , pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1)  # , pixel_norm=pixel_norm)
        if self.image_res >= 64:
            self.progression_64 = ConvBlock(in_channel, in_channel // 2, 3, 1)  # pixel_norm=pixel_norm)
        if self.image_res >= 128:
            self.progression_128 = ConvBlock(in_channel // 2, in_channel // 4, 3, 1)  # , pixel_norm=pixel_norm)
        if self.image_res >= 256:
            self.progression_256 = ConvBlock(in_channel // 4, in_channel // 4, 3, 1)  # , pixel_norm=pixel_norm)

        if self.image_res == 32:
            self.to_rgb_32 = nn.Conv2d(in_channel, 3, 1)
        if self.image_res == 64:
            self.to_rgb_64 = nn.Conv2d(in_channel // 2, 3, 1)
        if self.image_res == 128:
            self.to_rgb_128 = nn.Conv2d(in_channel // 4, 3, 1)
        if self.image_res == 256:
            self.to_rgb_256 = nn.Conv2d(in_channel // 4, 3, 1)

        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2)
        out = module(out)
        return out

    def output_simple(self, feat1, module1, alpha):
        out = module1(feat1)
        if self.tanh:
            return torch.sigmoid(out)
        return out

    def forward(self, input, step=6, alpha=0):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)
        out = self.progress(out_8, self.progression_16)

        resolutions = [32, 64, 128, 256]

        for res in resolutions:
            if self.image_res >= res:
                out = self.progress(out, getattr(self, f'progression_{res}'))
                if self.image_res == res:
                    return self.output_simple(out, getattr(self, f'to_rgb_{res}'), alpha)


def tv(x):
    return (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean() + (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()


def compare_diff(t1, t2):
    unique_elements1, counts1 = torch.unique(t1, return_counts=True)
    unique_elements2, counts2 = torch.unique(t2, return_counts=True)
    d1 = dict(zip(unique_elements1.tolist(), counts1.tolist()))
    d2 = dict(zip(unique_elements2.tolist(), counts2.tolist()))
    diff = 0
    for k, v in d2.items():
        if d1.get(k):
            diff += abs(d2[k] - d1[k])
        else:
            diff += d2[k]
    return diff


# 🌟 New
def ceil_enabled_res_if_needed(expected_res: int) -> int:
    """Method to ceil `img_res` to the maximum res that would be acceptable for Generator."""
    for res in [32, 64, 128, 256]:
        if res >= expected_res:
            return res
    return 256


# 🌟 New
def make_resizing_func(generated_res: int, expected_res: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Method to resize generated images."""
    if generated_res == expected_res:
        return lambda x: x
    else:
        return lambda x: TF.center_crop(x, output_size=[expected_res, expected_res]) 

# 🎯 Origin
# def FedLeak(client_grads, original_label, model, grad_diff_loss):
# 🌟 New
def FedLeak(client_grads, original_label, model, grad_diff_loss, img_res=128, need_upscaling=True, batch_size=16, num_iters=10000, plot_interval=1000, device=None):
    
    # 🎯 Origin
    # device = torch.device("cuda:0")
    # 🌟 New
    if device is None:
        device = torch.device("cuda:0")
    
    latent_vec = 128
    lr = 1e-2
    channel = 256
    # 🎯 Origin
    # batch_size = 16
    # num_iters = 10000
    # img_res = 128
    # netG = Generator(image_res=img_res, in_channel=channel).to(device)
    # 🌟 New
    gen_res = ceil_enabled_res_if_needed(expected_res=img_res)
    resizing_func = make_resizing_func(generated_res=gen_res, expected_res=img_res)
    netG = Generator(image_res=gen_res, in_channel=channel).to(device)
    
    noise = torch.randn(batch_size, latent_vec, device=device)
    dummy_label = torch.randn(original_label.size(0), 1000).to(device).requires_grad_(True)
    optimizerG = optim.AdamW(netG.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, 10001)
    cross_entropy = nn.CrossEntropyLoss().to(device)
    # indices = get_indices(client_grads, 15)
    median_filter = utils.MedianPool2d(kernel_size=5, stride=1, padding=2, same=False)
    # inferenced_labels = client_grads[-2].sum(dim=1).topk(k=original_label.size(0), largest=False)[1].long()
    # print(compare_diff(inferenced_labels, original_label))

    # 🎯 Origin
    # for iters in tqdm(range(num_iters)):
    # 🌟 New
    timestamp = time.time()
    for i in range(num_iters):
        optimizerG.zero_grad()

        # 🎯 Origin
        # reconstructed_imgs = netG(noise)
        # 🌟 New
        reconstructed_imgs = resizing_func(netG(noise))
        
        fake_output = model(reconstructed_imgs)
        dummy_loss = cross_entropy(fake_output, original_label)
        fake_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        ls = []
        for indice in range(len(client_grads)):
            ls.append(grad_diff_loss(client_grads[indice], fake_dy_dx[indice]))
        total_loss = sum(ls) + 1e-5 * tv(reconstructed_imgs)
        total_loss.backward()
        for params in netG.parameters():
            params.grad.sign_()
        ori_params = [params.data.clone() for params in netG.parameters()]
        ori_grads = [params.grad.data.clone() for params in netG.parameters()]

        optimizerG.step()
        optimizerG.zero_grad()

        # forward gradient
        # 🎯 Origin
        # reconstructed_imgs = netG(noise)
        # 🌟 New
        reconstructed_imgs = resizing_func(netG(noise))
        
        fake_output = model(reconstructed_imgs)
        dummy_loss = cross_entropy(fake_output, original_label)
        fake_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        ls = []
        for indice in range(len(client_grads)):
            ls.append(grad_diff_loss(client_grads[indice], fake_dy_dx[indice]))
        total_loss = sum(ls) + 1e-5 * tv(reconstructed_imgs)
        total_loss.backward()
        for params in netG.parameters():
            params.grad.sign_()
        forward_grads = [params.grad.data.clone() for params in netG.parameters()]
        for idx, params in enumerate(netG.parameters()):
            params.data = ori_params[idx]
            params.grad.data = 0.7 * ori_grads[idx] + 0.3 * forward_grads[idx]

        optimizerG.step()
        scheduler.step()
        
        # 🌟 New
        if i % plot_interval == 0:
            current_timestamp = time.time()
            print(
                f"| It: {i + 1} "
                f"| Loss: {total_loss.item():2.4f} "
                f"| Time: {current_timestamp - timestamp:6.2f}s |"
            )
            timestamp = current_timestamp
        

    # 🎯 Origin
    # return upscale(median_filter(reconstructed_imgs))
    # 🌟 New
    reconstructed_imgs = median_filter(reconstructed_imgs)
    if need_upscaling:
        reconstructed_imgs = upscale(reconstructed_imgs)
    return reconstructed_imgs

