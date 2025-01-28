#!/bin/bash
python extract_feature_cifar.py data/cifar10 outputs/densenet10_cifar10_train.pkl --img_list datalists/cifar10_train.txt --nn densenet --ind cifar10
python extract_feature_cifar.py data/cifar10 outputs/densenet10_cifar10_test.pkl --img_list datalists/cifar10_test.txt --nn densenet --ind cifar10
python extract_feature_cifar.py data/iSUN outputs/densenet10_iSUN.pkl --nn densenet --ind cifar10
python extract_feature_cifar.py data/LSUN outputs/densenet10_LSUN.pkl --nn densenet --ind cifar10
python extract_feature_cifar.py data/LSUN_resize outputs/densenet10_LSUN_resize.pkl --nn densenet --ind cifar10
python extract_feature_cifar.py data/places365 outputs/densenet10_places365.pkl --nn densenet --ind cifar10  --batch 128
python extract_feature_cifar.py data/svhn outputs/densenet10_svhn.pkl --nn densenet --ind cifar10
python extract_feature_cifar.py data/texture outputs/densenet10_texture.pkl --img_list datalists/texture.txt --nn densenet --ind cifar10
python extract_feature_cifar.py a b --fc_save_path outputs/densenet10_fc.pkl --nn densenet --ind cifar10