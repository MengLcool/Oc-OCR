exp_name='BERT_CTC_cpx'
train_data='ctwReCTS_train-Synth_v1-ic_40_correct-tianchi_140k-collect_img-ictc_rd_small_400k-ver_enhance_819_500k-cpx_400k'
train_ratio='0.1-0.1-0.1-0.1-0.2-0.1-0.1-0.2'
valid_data='/ai/local/menglc/v3_dataset/cpx_validate'

CUDA_VISIBLE_DEVICES=3 nohup \
python3 -u train.py \
 --experiment_name ${exp_name} \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BERT --Prediction CTC \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data /ai/local/menglc/v3_dataset/train \
 --valid_data ${valid_data} \
 --select_data  ${train_data}\
 --valInterval 500 \
 --batch_ratio ${train_ratio}\
 --manualSeed 2222 \
 --PAD \
 --batch_size 96 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 10000 \
 --character cn_v4.txt \
 --my_model 0 \
 --continue_model saved_models/${exp_name}/best_norm_ED.pth \
 --grad_clip 10 \
>>log/${exp_name}.log

