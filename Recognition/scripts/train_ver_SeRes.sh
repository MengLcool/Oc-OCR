CUDA_VISIBLE_DEVICES=5,6 nohup \
python3 -u train.py \
 --experiment_name vertical_SERes \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BiLSTM --Prediction Attn \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data result/ver_train \
 --valid_data result/ver_validate_v2 \
 --select_data noise_quasicrystal_v1_1100k-v2_700k \
 --valInterval 2000 \
 --batch_ratio 0.2-0.8 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model /home/menglc/ocr_new/saved_models/vertical_SERes/best_norm_ED.pth \
>result_ver_SEResNet.log

