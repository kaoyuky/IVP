# IVP
This is Pytorch implementation for "Rethinking Remote Sensing Pretrained Model: Instance-Aware Visual Prompting for Remote Sensing Scene Classification". If you have any questions, please contact kys220900680@hnu.edu.cn
## Overview
![image](picture/IVP.png)

## Dataset Preparation
We fine-tuned the pre-trained models on the UCM/AID/NWPU-RESISC45 dataset. For each dataset, we first merge all the images together, then split them into training and validation sets and recode their information in train_label.txt and valid_label.txt, respectively. an example of the format in train_label.txt is as follows:
```
P0960374.jpg dry_field 0
P0973343.jpg dry_field 0
P0235595.jpg dry_field 0
P0740591.jpg dry_field 0
P0099281.jpg dry_field 0
P0285964.jpg dry_field 0
...
```
Here, 0 is the training id of category for corresponded image.

## Training
* When iteratively fine-tuning the pre-trained Swin-T model on the AID dataset, the setting was (2:8) 5 times.
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 7777 main.py --dataset 'aid' --model 'swin' --ratio 28 --exp_num 5 --batch-size 64 --epochs 120 --img_size 224 --split 1 --lr 5e-4  --weight_decay 0.05 --gpu_num 1 --output Experiment_deep/checkpoint --pretrained /mnt/XXX/XXX//pretrained/rsp-swin-t-ckpt.pth --cfg configs/swin_tiny_patch4_window7_224.yaml
```

## Citation

   If you find our repo useful for your research, please consider citing our paper:
   ```bibtex
   @article{fang2023rethinking,
  title={Rethinking Remote Sensing Pretrained Model: Instance-Aware Visual Prompting for Remote Sensing Scene Classification},
  author={Fang, Leyuan and Kuang, Yang and Liu, Qiang and Yang, Yi and Yue, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--13},
  year={2023},
  publisher={IEEE}}
   ```

## References

The codes of Recognition part mainly from [An Empirical Study of Remote Sensing Pretraining](https://github.com/ViTAE-Transformer/RSP.git).
