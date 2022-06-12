import argparse
import os
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image

from models import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
args = parser.parse_args()

class Args:
    n_epochs = 15
    dataset_name = "celeb2manga"
    batch_size = 1
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    decay_epoch = 1
    n_cpu = 8
    img_height = 256
    img_width = 256
    channels = 3
    sample_interval = 2000
    checkpoint_interval = 1
    n_residual_blocks = 9
    lambda_cyc = 10.0
    lambda_id = 5.0


opt = Args()


cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)


if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    
# Load pretrained models
G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, args.epoch)))
G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, args.epoch)))


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
os.makedirs(f"images/galdx2manga/epoch{args.epoch}/", exist_ok=True)
# Image transformations
transforms_ = transforms.Compose([
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Transform from A (CelebA) to B (Manga)
directory = "images/galdx/"
G_BA.eval()
for j in range(3):
    # Load image
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = transforms_(image)
        if cuda:
            image = image.cuda()
        image.unsqueeze_(0)
        manga_galdyn = G_BA(image)
        save_image(manga_galdyn, f'images/galdx2manga/epoch{args.epoch}/{filename}_{j}.jpg')

