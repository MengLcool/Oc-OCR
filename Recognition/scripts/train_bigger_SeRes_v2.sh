CUDA_VISIBLE_DEVICES=1,2 nohup \
python3 -u train.py \
 --experiment_name bigger_train_SERes_v2 \
 --Transformation TPS --FeatureExtraction SEResNet --SequenceModeling BiLSTM --Prediction Attn \
 --num_fiducial 20 \
 --input_channel 1 \
 --output_channel 512 \
 --hidden_size 256 \
 --adam --lr 0.001 \
 --train_data result/bigger_train \
 --valid_data result/cut_hor_validate \
 --select_data enhance_cn-enhance_illegal-enhance_real-ic_40_correct-tianchi_140k-enhance_genreal_300k-ictq_sexy_500k-v2_4100k-ictc_rd_small_400k \
 --valInterval 2000 \
 --batch_ratio 0.025-0.025-0.05-0.1-0.1-0.1-0.2-0.2-0.2 \
 --manualSeed 2222 \
 --PAD \
 --batch_size 192 \
 --batch_max_length 35 \
 --imgH 32 --imgW 256 \
 --minC 4000 \
 --character cn_bigger.txt \
 --continue_model /home/menglc/ocr_new/saved_models/bigger_train_SERes_v2/best_norm_ED.pth \
>>result_bigger_SEResNet_v2.log

