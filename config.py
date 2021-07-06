import sys 

from .Craft.craft_cfg import CraftConfig
from .Recognition.model_cfg import RecognitionConfig

# /home/tione/notebook/VideoStructuring/feature_extract/mlc/text_detection_recognition

g_craft_cfg = CraftConfig(
                #saved_model= '/home/tione/notebook/VideoStructuring/feature_extract/mlc/craft_mlt_25k.pth',
                saved_model= 'pretrain_models/craft_mlt_25k.pth',
                text_threshold=0.3, 
                low_text=0.3, 
                link_threshold=0.3, 
                canvas_size=1280, 
                poly= False, 
                mag_ratio= 1)

g_BERT_CTC_cpx_config = RecognitionConfig(
        #saved_model = '/home/tione/notebook/VideoStructuring/feature_extract/mlc/bert_ctc_cpx_v2.pth',
        saved_model = 'pretrain_models/bert_ctc_cpx_v2.pth',
        #character = '/home/tione/notebook/VideoStructuring/feature_extract/mlc/cn_v4.txt',
        character = 'pretrain_models/cn_v4.txt',
        batch_max_length = 35,
        imgH = 32,
        imgW = 256,
        minC = 10000,
        PAD = True,
        Transformation = 'TPS',
        FeatureExtraction = 'SEResNet',
        SequenceModeling = 'BERT',
        Prediction = 'CTC',
        num_fiducial = 20,
        input_channel = 1,
        output_channel = 512,
        hidden_size = 256
    )
