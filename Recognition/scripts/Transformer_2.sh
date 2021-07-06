CUDA_VISIBLE_DEVICES=4,7 nohup python3 -u train.py \
 --experiment_name Transformer_2 \
 --input_channel 3 \
 --lr 1 \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-v2_4100k-v2_700k-ver_enhance_819_500k-collect_img\
 --valInterval 500 \
 --batch_ratio 0.1-0.25-0.15-0.1-0.1-0.1-0.2 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/validate \
 --manualSeed 2222 \
 --PAD \
 --batch_size 128 \
 --batch_max_length 64 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --my_model 1 \
 --transformer_model 1 \
 --output_channel 512 \
>>log/Transformer_2.log

