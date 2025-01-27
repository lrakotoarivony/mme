#!/bin/bash
./extract_feature_bit.py data/imagenet outputs/bit_train_200k.pkl --img_list datalists/imagenet2012_train_random_200k_reformatted.txt
./extract_feature_bit.py data/imagenet outputs/bit_imagenet_val.pkl --img_list datalists/imagenet2012_val_list.txt
./extract_feature_bit.py data/texture outputs/bit_texture.pkl --img_list datalists/texture.txt
./extract_feature_bit.py data/inaturalist outputs/bit_inaturalist.pkl
./extract_feature_bit.py data/places outputs/bit_places.pkl
./extract_feature_bit.py data/sun outputs/bit_sun.pkl
./extract_feature_bit.py data/imagenet_o outputs/bit_imagenet_o.pkl
./extract_feature_bit.py a b --fc_save_path outputs/bit_fc.pkl