#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py --maxdisp 256 \
				   --rgb_datapath ../dataset/SceneFlow/ \
				   --aug_datapath ../dataset_3DAug/SceneFlow/near_focus/3/ \
				   --epochs 20 \
				   --loadmodel ./aug_model_save/best_model_dof.tar \
				   --savemodel ./aug_model_save/
