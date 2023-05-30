import torch
import torchaudio.transforms as tran

class Conversations(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.transforms = tran.MelSpectrogram()


    def __getitem__(self):
        pass

    def __len__(self):
        pass

