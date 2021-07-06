CUDA_VISIBLE_DEVICES=3,4 nohup \
python3 -u train.py \
 --experiment_name my_HV_CTC_v2 \
 --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 3 \
 --output_channel 2048 \
 --hidden_size 512 \
 --lr 0.001 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/validate \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-v2_4100k-v2_700k-ver_enhance_819_500k\
 --valInterval 1000 \
 --batch_ratio 0.1-0.25-0.15-0.1-0.2-0.2 \
 --manualSeed 2222 \
 --rgb \
 --PAD \
 --batch_size 394 \
 --batch_max_length 35 \
 --imgH 32 --imgW 196 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model saved_models/my_HV_CTC_v2/best_norm_ED.pth \
>>log/my_HV_SERes_CTC_2.log

