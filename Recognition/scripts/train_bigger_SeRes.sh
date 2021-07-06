CUDA_VISIBLE_DEVICES=2 nohup \
python3 -u train.py \
 --experiment_name bigger_train_SERes_v2 \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BiLSTM --Prediction Attn \
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
 --batch_size 64 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model /home/menglc/ocr_new/saved_models/bigger_train_SERes/best_accuracy.pth \
>result_bigger_SEResNet_v2.log
