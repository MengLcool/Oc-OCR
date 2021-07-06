CUDA_VISIBLE_DEVICES=1 nohup \
python3 -u train.py \
 --experiment_name bigger_set_lstm \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BiLSTM --Prediction Attn \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data result/train \
 --valid_data result/validate_bigger \
 --select_data Hor_bg-Hor_no \
 --valInterval 5000 \
 --batch_ratio 0.9-0.1 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 64 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
>result_bigger.log
