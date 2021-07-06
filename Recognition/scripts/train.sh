CUDA_VISIBLE_DEVICES=2 nohup \
python3 -u train.py \
--experiment_name small_set \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--num_fiducial 20 \
--input_channel 1 \
--output_channel 512 \
--hidden_size 256 \
--adam --lr 0.0001 \
--train_data result/train \
--valid_data result/validate \
--select_data plain \
--batch_ratio 1 \
--manualSeed 2222 \
--PAD \
--batch_size 64 \
--batch_max_length 35 \
--imgH 32 --imgW 256 \
--character cn.txt \
--continue_model saved_models/log/best_accuracy.pth \
>result1.log
