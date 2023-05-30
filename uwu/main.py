import torch
from .dataset import Conversations
from .models import Generator

class Uwu:
    def __init__(self,config):

        self.num_epoch = config.num_epoch
        self.lr = config.lr
        self.device = config.device
        self.dataset = Conversations()
        self.model = Generator().to(self.device)


    def generate(self,texts):
        audio  = self.model(texts)
        return audio

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        

        for epoch in range(self.num_epoch):
            for i,(x,y) in enumerate(self.dataset):
                optimizer.zero_grad()

            print(f"Epoch {epoch} done...")

    def load_pretrained(self,name=''):
        pass