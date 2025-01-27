#!/bin/bash
./extract_feature_vit.py data/imagenet outputs/vit_train_200k.pkl --img_list datalists/imagenet2012_train_random_200k_reformatted.txt
./extract_feature_vit.py data/texture outputs/vit_texture.pkl --img_list datalists/texture.txt
./extract_feature_vit.py data/cifar10 outputs/vit_cifar10_train.pkl --img_list datalists/cifar10_train.txt
./extract_feature_vit.py data/cifar10 outputs/vit_cifar10_test.pkl --img_list datalists/cifar10_test.txt
./extract_feature_vit.py data/imagenet_o outputs/vit_imagenet_o.pkl
