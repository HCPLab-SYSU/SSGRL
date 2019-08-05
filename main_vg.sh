#!/bin/bash

#LOG="log/ResNet101-baseline-448-Adam-1e-5-bs16.txt"
LOG="log/finetune-81.8-`date +'%Y-%m-%d_%H-%M-%S'`"
#exec &> >(tee -a "$LOG")

# usage:
#   ./main.sh [post(any content to record the conducted experiment)]
#LOG="log/bcnn.`date +'%Y-%m-%d_%H-%M-%S'`"
#exec &> >(tee -a "$LOG")

train_data_dir='./data/VG/VG_100K'
train_list='./data/VG/train_list_500.txt'
test_data_dir='./data/VG/VG_100K'
test_list='./data/VG/test_list_500.txt'
train_label='./data/VG/vg_category_500_labels_index.json'
test_label='./data/VG/vg_category_500_labels_index.json'

graph_file='./data/VG/graph_500_norm.npy'
word_file='./data/VG/vg_500_vector.npy'
#word_file='../data/coco_random_shuffle.npy'
batch_size=4
epochs=200
learning_rate=1e-5
momentum=0.9
weight_decay=0
num_classes=500
pretrained=1
pretrain_model='./pretrain_model/resnet101.pth.tar'
#input parameter 
crop_size=576
scale_size=640

#number of data loading workers
workers=2
#manual epoch number (useful on restarts)
start_epoch=0
#epoch number to decend lr
step_epoch=1516541
#print frequency (default: 10)
print_freq=500
#path to latest checkpoint (default: none)
#resume="model_best_vgg_pretrain_bk.pth.tar"
#resume="backup/86.26.pth.tar"
#evaluate mode
evaluate=false
extra_cmd=""
if $evaluate  
then 
  extra_cmd="$extra_cmd --evaluate"
fi
# resume is not none
if [ -n "$resume" ]; 
then
  extra_cmd="$extra_cmd --resume $resume"
fi


# use single gpu (eg,gpu 0) to trian:
#     CUDA_VISIBLE_DEVICES=0 
# use multiple gpu (eg,gpu 0 and 1) to train
#     CUDA_VISIBLE_DEVICES=0,1  
CUDA_VISIBLE_DEVICES=$1 python main.py \
    ${train_data_dir} \
    ${test_data_dir} \
    ${train_list} \
    ${test_list}  \
    -b ${batch_size} \
    -train_label ${train_label} \
    -test_label ${test_label} \
    -graph_file ${graph_file} \
    -word_file ${word_file} \
    -j ${workers} \
    --epochs ${epochs} \
    --start-epoch  ${start_epoch} \
    --batch-size ${batch_size} \
    --learning-rate ${learning_rate} \
    --momentum ${momentum} \
    --weight-decay ${weight_decay} \
    --crop_size ${crop_size} \
    --scale_size ${scale_size} \
    --step_epoch ${step_epoch} \
    --print_freq ${print_freq} \
    --pretrained ${pretrained} \
    --pretrain_model ${pretrain_model} \
    --num_classes ${num_classes} \
    --post $2\
    ${extra_cmd}
