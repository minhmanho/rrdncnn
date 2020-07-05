#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP32
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP37
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP42
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP47
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn_v2.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP32
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn_v2.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP37
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn_v2.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP42
CUDA_VISIBLE_DEVICES=0 python eval.py --vin ./data/class_e/ --ckpt ./models/rrdncnn_v2.pth.tar --size 360x640 --decoder_folder DLR_LDP_QP47
