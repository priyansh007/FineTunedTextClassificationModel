import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, input_ids, attention_masks, y_labels):
        super().__init__()
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.y_labels = y_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        ans = {
              'input_ids': self.input_ids[index],
              'attention_mask': self.attention_masks[index],
              'labels': torch.tensor([self.y_labels[index]], dtype=torch.long)
          }
        return ans