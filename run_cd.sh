#!/usr/bin/env bash



gpus=2
checkpoint_root=checkpoints
data_name=LEVIR
dataset=CDDataset

img_size=256
batch_size=16
lr=2e-4
max_epochs=200

net_G=CSST_Siam_RPT

loss=ce
n_class=2
optimizer=adamw
lr_policy=linear

pretrain=/data/resnet18-5c106cde.pth

split=train
split_val=val
#split_val=test  # for CMU data set

project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --loss ${loss} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --n_class ${n_class} --dataset ${dataset} --optimizer ${optimizer} --pretrain ${pretrain}
