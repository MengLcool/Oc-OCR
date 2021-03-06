CUDA_VISIBLE_DEVICES=5,6 nohup \
python3 -u train.py \
 --experiment_name BERT_CTC_new \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BERT --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/ctwRes_validate \
 --select_data  ic_40_correct-tianchi_140k-ictc_rd_small_400k-v2_4100k-v2_700k-ver_enhance_819_500k-collect_img\
 --valInterval 500 \
 --batch_ratio 0.1-0.25-0.15-0.1-0.1-0.1-0.2 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 5000 \
 --character cn_bigger.txt \
 --my_model 0 \
>>log/BERT_CTC_new.log

