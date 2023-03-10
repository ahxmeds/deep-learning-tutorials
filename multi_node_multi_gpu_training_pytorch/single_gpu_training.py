# code running training on a single GPU 


#%%
import torch 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F
import torch.nn as nn

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every_epoch: int
    ) -> None:
        self.gpu_id = gpu_id,
        self.model = model,
        self.dataloader_train = dataloader_train,
        self.optimizer = optimizer, 
        self.save_every_epoch = save_every_epoch
    
    def run_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def run_epoch(self, epoch: int):
        batch_size = len(next(iter(self.dataloader_train))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {batch_size} | Steps: {len(self.dataloader_train)}")
        for inputs, targets in self.dataloader_train:
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.run_batch(inputs, targets)
    
    def save_checkpoint(self, epoch):
        checkpnt = self.model.state_dict()
        PATH = 'checkpoint.pth'
        torch.save(checkpnt, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs:int):
        for epoch in range(max_epochs):
            self.run_epoch(epoch)
            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch)

#%%
class TrainDataset(Dataset):
    def __init__(self, )


def get_training_objects():
    dataset_train = TrainDataset() # Dataset class using a randomly generated toy dataset

    model = nn.Sequential(
        nn.Linear(100, 20, bias=True),
        nn.ReLU(),
        nn.Linear(20, 1, bias=True),
        nn.ReLU()
    )# toy model with 2 dense layers; just for experimentation
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    return dataset_train, model, optimizer


    




# %%
