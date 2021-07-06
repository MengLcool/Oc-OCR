CUDA_VISIBLE_DEVICES=1 nohup \
python3 -u train.py \
 --experiment_name BERT_CTC \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BERT --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data /ai/local/menglc/v3_dataset/ctwRes_validate \
 --select_data  ctwReCTS_train-Synth_v1-ic_40_correct-tianchi_140k-collect_img-ictc_rd_small_400k-ver_enhance_819_500k\
 --valInterval 500 \
 --batch_ratio 0.2-0.2-0.1-0.1-0.2-0.1-0.1 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 128 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 5000 \
 --character new_dict.txt \
 --my_model 0 \
 --continue_model saved_models/BERT_CTC/best_norm_ED.pth \
>>log/BERT_CTC.log

