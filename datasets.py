import torch
import torch.utils.data as data

def add_arguments(parser):
    parser.add_argument('--dataset_n_train_samples', type=int, default=1000)
    parser.add_argument('--dataset_seed', type=int, default=0)

def get_dataset(args):
    x = torch.normal(mean=0., std=1., size=(args.dataset_n_train_samples, args.model_inp_dim))
    y = torch.where(x.sum(axis=1)>0, 0, 1)
    dataset = data.TensorDataset(x, y)
    data_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
    return data_loader