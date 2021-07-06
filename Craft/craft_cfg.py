
class CraftConfig():

    def __init__(self, saved_model,text_threshold=0.3, low_text=0.3, link_threshold=0.3, canvas_size=1080, poly= False, mag_ratio= 1.5):
        
        self.saved_model = saved_model
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.poly = poly
        self.mag_ratio = mag_ratio
