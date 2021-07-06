CUDA_VISIBLE_DEVICES=3,4 nohup \
python3 -u train.py \
 --experiment_name my_HV_CTC \
 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --lr 0.1 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/validate \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-Synth_v1-ver_enhance_819_500k-collect_img\
 --valInterval 500 \
 --batch_ratio 0.1-0.15-0.15-0.3-0.1-0.2 \
 --manualSeed 2222 \
 --rgb \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model saved_models/my_HV_CTC/best_norm_ED.pth \
>>log/my_HV_SERes_CTC.log

