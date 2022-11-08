import torch.utils.data as data

from PIL import Image
import os
import os.path

import random
from random import choice
import torchvision.transforms as trans
from skimage import feature
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from skimage import filters
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def accimage_loader(path):
    print("can't find acc image loader")
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



# define the own imagefolder   from code for torchvision.datasets.folder
class ZJFolder(data.Dataset):

    def __init__(self, root, opt, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)

        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.opt = opt
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        # self.transform = trans.Compose([
        #     trans.RandomCrop(32, padding=4),
        #     trans.RandomHorizontalFlip()])

        self.target_transform = target_transform
        self.loader = loader
        self.len=len(imgs)
    def __getitem__(self, index):

        opt = self.opt
        path, target = self.imgs[index]
        img = self.loader(path)
        # img = self.transform(img)

        random_R = random.randint(1, 5) * 40
        random_G = random.randint(1, 5) * 40
        random_B = random.randint(1, 5) * 40
        color = torch.ones((3))
        color[0] = random_R
        color[1] = random_G
        color[2] = random_B

        # if opt.rot and random.randint(0,1) :
        #     degree = random.uniform(-30, 30)
        # elif opt.rot15 and random.randint(0,1) :
        #     degree = random.uniform(-15, 15)
        # else:
        #     degree = 0
        degree = 0

        theta = math.pi / 180 * degree
        rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                            [math.sin(theta), math.cos(theta), 0],])
                                            # [0, 0, 1]])
        rotation_matrix = rotation_matrix



        img_np = np.array(img)
        img_t = torch.from_numpy(img_np.transpose((2, 0, 1)))
        img_t =  img_t.float().div(255)
        img_t_torot = img_t.unsqueeze(0)   #只有此处是四维张量 ,ZJrot 输入要求

        img_gray_np = 0.299 * img_np[:,:,0] +  0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        mask = filters.sobel(img_gray_np)
        mask = mask > 28 #23.04
        mask = mask.astype(np.uint8)
        mask = mask[:,:,np.newaxis]

        edge_np = np.ones_like(img_np)
        edge_np[:,:,0:1] = random_R * mask
        edge_np[:,:,1:2] = random_G * mask 
        edge_np[:,:,2:3] = random_B * mask 
        edge_t = torch.from_numpy(edge_np.transpose((2, 0, 1)))
        edge_t = edge_t.float().div(255)

        mask_np =np.repeat(mask,3,axis=2)
        mask_t = torch.from_numpy(mask_np.transpose((2, 0, 1))).float()

        ################################  ROT  ##############################################

        img_rot_t = ZJ_ROT(img_t_torot,degree)  #img_B_t 必须是4维张量
        img_rot_t = img_rot_t.squeeze(0)


        return img_t, edge_t, mask_t, color, degree, rotation_matrix, img_rot_t

    def __len__(self):
        return self.len

# def normalize():
#     return  transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])

def ZJ_ROT(ori, degree):
    # if degree == 0:
    #     x=ori
    # else:
    pad_dims = [2, 2, 2, 2]
    x = F.pad(ori, pad_dims, 'constant', value=0)
    x = Rotate(degree, interp='bilinear')(x)
    x = x[:, :, 2:34, 2:34]

    return x


class Rotate(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
#  self 4D tensor
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy

    # @torchsnooper.snoop()
    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp] * len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                             [math.sin(theta), math.cos(theta), 0], ])
        # [0, 0, 1]])

        rotation_matrix = rotation_matrix.unsqueeze(0)
        # rotation_matrix = rotation_matrix.repeat(opt.batchSize, 1, 1 ) #此处的batchsize怎么改
        rotation_matrix = rotation_matrix.repeat(1, 1, 1)

        # print('DATA RRRRRRRRRRRRR',rotation_matrix)

        if self.lazy:
            return rotation_matrix
        else:

            for idx, _input in enumerate(inputs):

                outputs = _input
                for i in range(3):
                    input_tf = F_batch_affine2d(_input[:, i:i + 1, :, :], rotation_matrix, center=True)
                    outputs[:, i:i + 1, :, :] = input_tf

            return outputs


def F_batch_affine2d(x, matrix, center=True):
    if matrix.dim() == 2:
        matrix = matrix.view(-1, 2, 3)

    A_batch = matrix[:, :, :2]
    b_batch = matrix[:, :, 2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = torch_iterproduct(x.size(2), x.size(3))
    coords = Variable(_coords.unsqueeze(0).repeat(x.size(0), 1, 1).float(),
                      requires_grad=False)

    if center:
        # shift the coordinates so center is the origin
        coords[:, :, 0] = coords[:, :, 0] - (x.size(2) / 2. + 0.5)
        coords[:, :, 1] = coords[:, :, 1] - (x.size(3) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(2) / 2. + 0.5)
        new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(3) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_batch_bilinear_interp2d(x, new_coords)

    return x_transformed


def F_batch_bilinear_interp2d(input, coords):
    """
    input : torch.Tensor
        size = (N,H,W,C)
    coords : torch.Tensor
        size = (N,H*W*C,2)
    """
    x = torch.clamp(coords[:, :, 0], 0, input.size(2) - 2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:, :, 1], 0, input.size(3) - 2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[2]).long()
    x1_ix = x1.mul(stride[2]).long()
    y0_ix = y0.mul(stride[3]).long()
    y1_ix = y1.mul(stride[3]).long()

    input_flat = input.view(input.size(0), -1).contiguous()

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix).detach())
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix).detach())
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix).detach())
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix).detach())

    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def torch_iterproduct(*args):
    return torch.from_numpy(np.indices(args).reshape((len(args), -1)).T)


def main():
    # pth = "/data-x/g12/zhangjie/nips/datasets/cifar10png/test"
    # # dataset = ZJFolder(pth)
    # # loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    # # for i, data in enumerate(loader, 0):
    # #     img, edge = data
    # #     save_image(img,"./try/%02d_img.png"%i)
    # #     save_image(edge,"./try/%02dedge.png"%i)

    print("ZJ_Dataset")

if __name__ == '__main__':
    main()