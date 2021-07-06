 
class RecognitionConfig():

    def __init__(self, saved_model, batch_max_length, imgH, imgW, character, minC, PAD, 
                Transformation, FeatureExtraction, SequenceModeling, Prediction,
                num_fiducial, input_channel, output_channel, hidden_size):
            
        self.saved_model = saved_model 
        self.batch_max_length = batch_max_length 
        self.imgH = imgH 
        self.imgW = imgW 
        self.character = character 
        self.minC = minC 
        self.PAD = PAD 
        self.Transformation = Transformation 
        self.FeatureExtraction = FeatureExtraction 
        self.SequenceModeling = SequenceModeling 
        self.Prediction = Prediction 
        self.num_fiducial = num_fiducial 
        self.input_channel = input_channel 
        self.output_channel = output_channel 
        self.hidden_size = hidden_size


g_BERT_CTC_config = RecognitionConfig(
        saved_model = 'saved_models/BERT_CTC/best_norm_ED.pth',
        batch_max_length = 35,
        imgH = 32,
        imgW = 256,
        character = 'new_dict.txt',
        minC = 5000,
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

g_BERT_CTC_cpx_config = RecognitionConfig(
        saved_model = '/home/menglc/ocr_new/saved_models/BERT_CTC_cpx_v2/best_norm_ED.pth',
        batch_max_length = 35,
        imgH = 32,
        imgW = 256,
        character = 'cn_v4.txt',
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


g_BERT_Attn_config = RecognitionConfig(
        saved_model = 'saved_models/BERT_Attn/best_norm_ED.pth',
        batch_max_length = 35,
        imgH = 32,
        imgW = 256,
        character = 'new_dict.txt',
        minC = 5000,
        PAD = True,
        Transformation = 'TPS',
        FeatureExtraction = 'SEResNet',
        SequenceModeling = 'BERT',
        Prediction = 'Attn',
        num_fiducial = 20,
        input_channel = 1,
        output_channel = 512,
        hidden_size = 256
    )


g_BERT_Attn_cpx_config = RecognitionConfig(
        saved_model = 'saved_models/BERT_Attn_cpx/best_norm_ED.pth',
        batch_max_length = 35,
        imgH = 32,
        imgW = 256,
        character = 'cn_v4.txt',
        minC = 10000,
        PAD = True,
        Transformation = 'TPS',
        FeatureExtraction = 'SEResNet',
        SequenceModeling = 'BERT',
        Prediction = 'Attn',
        num_fiducial = 20,
        input_channel = 1,
        output_channel = 512,
        hidden_size = 256
    )
