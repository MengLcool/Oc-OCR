CUDA_VISIBLE_DEVICES=4 nohup \
python3 -u train.py \
 --experiment_name pre_bigger \
 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data result/bigger_train \
 --valid_data result/bigger_validate \
 --select_data v1-enhance_cn \
 --valInterval 5000 \
 --batch_ratio 0.3-0.7 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 100 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model /home/menglc/ocr_new/saved_models/pre_bigger/best_norm_ED.pth \
>log/pretrain_bigger.log
