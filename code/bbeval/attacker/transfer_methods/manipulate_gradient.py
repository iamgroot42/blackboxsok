import os
import torch
import torch.nn.functional as F
import numpy as np

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
    
def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, kern_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
    return x

def torch_staircase_sign(noise, n):
    noise_staircase = torch.zeros(size=noise.shape).cuda()
    sign = torch.sign(noise).cuda()
    temp_noise = noise.cuda()
    abs_noise = abs(noise)
    base = n / 100
    percentile = []
    for i in np.arange(n, 100.1, n):
        percentile.append(i / 100.0)
    medium_now = torch.quantile(abs_noise.reshape(len(abs_noise), -1), q = torch.tensor(percentile, dtype=torch.float32).cuda(), dim = 1, keepdim = True).unsqueeze(2).unsqueeze(3)

    for j in range(len(medium_now)):
        # print(temp_noise.shape)
        # print(medium_now[j].shape)
        update = sign * (abs(temp_noise) <= medium_now[j]) * (base + 2 * base * j)
        noise_staircase += update
        temp_noise += update * 1e5

    return noise_staircase


