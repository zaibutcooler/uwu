import torch
from huggingface_hub import login

from .dataset import Trainset
from .model import AudioGenerator
from .config import Config
from .utils import dummy_logger


class Uwu:
    def __init__(self, config: Config):
        self.config = config
        self.generator = AudioGenerator(config)
        self.num_params = self.get_num_params()

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        dummy_logger(f"parameter count -> {n_params}")
        return n_params

    def save_pretrained(self, name="uwu"):
        self.model.save_pretrained(name)
        self.model.push_to_hub(name)
        dummy_logger("Successfully saved the pretrainied")

    def load_pretrained(self, url="zaibutcooler/uwu"):
        self.model = self.gpt.from_pretrained(url)
        dummy_logger("Successfully loaded the pretrained")

    def huggingface_login(self, token):
        assert token is not None
        login(token=token)
        dummy_logger("Logged in successfully")

    def generate_audio(self, texts):
        audio = self.generator(texts)
        return audio

    def train(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(4):
            for i, (x, y) in enumerate([]):
                optimizer.zero_grad()

            print(f"Epoch {epoch} done...")
