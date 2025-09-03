import argparse
import pickle
from os.path import dirname

import mmengine
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn

from list_dataset import ImageFilelist
from mmpretrain.apis import init_model

from densenet import DenseNet3
from wideresnet import WideResNet
from svhn_loader import SVHN
#from models import densenet

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('data_root', help='Path to data')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('--nn', default="densenet", type=str, help='neural network name and training set')
    parser.add_argument('--ind', default="cifar10", type=str, help='neural network name and training set')
    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Path to data')
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)

    if args.nn == 'densenet':
        if args.ind == 'cifar10':
            pretrained_model = 'checkpoints/densenet10.pth'
            model = DenseNet3(100, 10)
        else:
            pretrained_model = 'checkpoints/densenet100.pth'
            model = DenseNet3(100, 100)

    
    transform = transforms.Compose([transforms.Resize(32),
            transforms.CenterCrop(32), 
            transforms.ToTensor(), 
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
            
    model = torch.load(pretrained_model, map_location='cpu')
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    #model.load_state_dict(torch.load(pretrained_model))
    model.cuda().eval()

    if args.fc_save_path is not None:
        mmengine.mkdir_or_exist(dirname(args.fc_save_path))
        w = model.fc.weight.cpu().detach().squeeze().numpy()
        b = model.fc.bias.cpu().detach().squeeze().numpy()
        with open(args.fc_save_path, 'wb') as f:
            pickle.dump([w, b], f)
        return
    
    if args.img_list is not None:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    elif 'svhn' in args.data_root:
        dataset = SVHN(args.data_root, transform=transform, split='test', download=False)
    else:
        dataset = torchvision.datasets.ImageFolder(args.data_root, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    
    features = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.cuda()
            feat_batch = model(x).cpu().numpy()
            features.append(feat_batch)

    features = np.concatenate(features, axis=0)
    print(features.shape)

    mmengine.mkdir_or_exist(dirname(args.out_file))
    with open(args.out_file, 'wb') as f:
        pickle.dump(features, f)



if __name__ == '__main__':
    main()




