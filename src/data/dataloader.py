import torch
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2


# preprocesses csv rows
def preprocess_row(row):  
    return torch.tensor([float(i) for i in row])


# preprocesses a batch of csv rows
def preprocess_batch(batch):

    batch_tensor = torch.stack(batch)
    
    return {
        "ID": batch_tensor[:, 0],
        "feats": batch_tensor[:, 1:-1], # include treatment
        "treatment": batch_tensor[:, -2],
        "outcome": batch_tensor[:, -1],
    }


# datapipe iterator object
def build_dataloader(PATH, batch_size, total_length=9659):
    
    # find and open csv file
    datapipe = dp.iter.IterableWrapper([PATH])
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    
    # parse rows from csv
    datapipe = datapipe.parse_csv(delimiter=',', skip_lines=1)
    
    # preprocess rows (to tensor)
    datapipe = datapipe.map(preprocess_row)
    
    # split into train / test set
    train_dp, test_dp = datapipe.random_split(
        total_length = total_length, 
        weights = {"train": 0.8, "test": 0.2}, 
        seed = 0)
    
    # shuffle train set
    train_dp = train_dp.shuffle().set_seed(0)
    
    # batch 
    train_dp_batch = train_dp.batch(batch_size=batch_size)
    test_dp_batch = test_dp.batch(batch_size=batch_size) # r2 score requires >1 sample
    
    # preprocess batches
    train_dp_batch = train_dp_batch.map(preprocess_batch)
    test_dp_batch = test_dp_batch.map(preprocess_batch)
    
    # to dataloader object
    train_loader = DataLoader2(train_dp_batch)
    test_loader = DataLoader2(test_dp_batch)
    
    # for other evaluation
    test_dp_unbatched = test_dp.batch(batch_size=1)
    test_dp_unbatched = test_dp_unbatched.map(preprocess_batch)
    test_loader_unbatched = DataLoader2(test_dp_unbatched)
    
    return train_loader, test_loader, test_loader_unbatched