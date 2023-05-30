import torch
from .dataset import Conversations
from .models import AudioGenerator,TextPredictor

class Uwu:
    def __init__(self,config):

        self.num_epoch = config.num_epoch
        self.lr = config.lr
        self.device = config.device
        self.dataset = Conversations()
        self.audio_generator = AudioGenerator().to(self.device)
        self.text_predictor = TextPredictor().to(self.device)


    def generate_audio(self,texts):
        audio  = self.audio_generator(texts)
        return audio
    
    def predict_text(self,audio):
        pass

    def train(self):
        optimizer = torch.optim.Adam(self.audio_generator.parameters(),lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        

        for epoch in range(self.num_epoch):
            for i,(x,y) in enumerate(self.dataset):
                optimizer.zero_grad()

            print(f"Epoch {epoch} done...")

    def download_pretrained(self,name=''):
        pass

    