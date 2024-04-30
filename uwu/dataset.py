from torch.utils.data import Dataset
from datasets import load_dataset
from .utils import dummy_logger


class Trainset(Dataset):
    def __init__(self):
        self.audio = None
        self.texts = None

        self.transforms = []

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return []

    def _load_dataset(self, url="zaibutcooler/conversations"):
        loaded_dataset = load_dataset(url)

        dummy_logger("Successfully loaded the dataset")
