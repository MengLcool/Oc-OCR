CUDA_VISIBLE_DEVICES=1 nohup python3 -u train.py \
 --experiment_name Bert_Full \
 --input_channel 3 \
 --lr 0.001 \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-Synth_v1-ver_enhance_819_500k-collect_img\
 --valInterval 500 \
 --batch_ratio 0.1-0.15-0.15-0.3-0.1-0.2 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/validate \
 --manualSeed 2333 \
 --PAD \
 --batch_size 64 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 5000 \
 --character cn_bigger.txt \
 --my_model 1 \
 --transformer_model 1 \
 --output_channel 128 \
 --continue_model saved_models/Bert_Full/best_norm_ED.pth \
>>log/Bert_6.log

