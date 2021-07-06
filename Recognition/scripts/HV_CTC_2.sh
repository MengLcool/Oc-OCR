CUDA_VISIBLE_DEVICES=1,2 nohup \
python3 -u train.py \
 --experiment_name HV_CTC_2 \
 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/validate \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-v2_4100k-v2_700k-ver_enhance_819_500k\
 --valInterval 1000 \
 --batch_ratio 0.1-0.25-0.15-0.1-0.2-0.2 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --my_model 0 \
>>log/HV_Res_CTC_2.log

