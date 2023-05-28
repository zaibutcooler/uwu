import torch
from dataset import Conversations
from models import Generator

class Uwu:
    def __init__(self,num_epoch=20,lr=0.01,):

        self.num_epoch = num_epoch
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = Conversations()
        self.model = Generator().to(self.device)


    def generate(self,texts):
        pass

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        for epoch in range(self.num_epoch):
            for i,(x,y) in enumerate(self.dataset):
                optimizer.zero_grad()

            print(f"Epoch {epoch} done...")
