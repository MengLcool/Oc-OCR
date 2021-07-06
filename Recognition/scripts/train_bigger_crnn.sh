CUDA_VISIBLE_DEVICES=2 nohup \
python3 -u train.py \
 --experiment_name bigger_crnn \
 --Transformation TPS --FeatureExtraction RCNN --SequenceModeling BiLSTM --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 1024 \
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
>crnn_v2.log 
