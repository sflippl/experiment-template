import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
import torchmetrics as tm

def add_arguments(parser):
    parser.add_argument('--model_inp_dim', type=int, required=True)
    parser.add_argument('--model_outp_dim', type=int, required=True)
    parser.add_argument('--model_hidden_dims', type=int, nargs='+', default=None)
    parser.add_argument('--model_optimizer_lr', type=float, default=1e-4)
    parser.add_argument('--model_optimizer_type', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--model_seed', type=int, default=0)

def get_model(args):
    return DenseNetModule(args)

class DenseNetModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = DenseNet(args)
        self.args = args
        self.save_hyperparameters(args)
    
    # This specifies what is done during each training and validation step.
    def step(self, batch, batch_idx, split):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        self.log(f'{split}_loss', loss)
        acc = tm.functional.accuracy(task="multiclass", num_classes=self.args.model_outp_dim, preds=pred, target=y)
        self.log(f'{split}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, 'test')

    # With Pytorch Lightning, we specify the optimizers as a part of the model.
    def configure_optimizers(self):
        if self.args.model_optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.model_optimizer_lr, momentum=self.args.model_optimizer_momentum)
        elif self.args.model_optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.model_optimizer_lr)

# A simple dense neural network
class DenseNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(seed=args.model_seed)
        hdims = args.model_hidden_dims or [1000]
        L = []
        for i, (_in, _out) in enumerate(zip([args.model_inp_dim]+hdims[:-1], hdims)):
            linear = nn.Linear(_in, _out, bias=True)
            nn.init.kaiming_normal_(linear.weight)
            L.append(linear)
            L.append(nn.ReLU())
        self.features = nn.Sequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = args.model_inp_dim
        self.readout = nn.Linear(_in, args.model_outp_dim, bias=True)
        nn.init.kaiming_normal_(self.readout.weight)
    
    def forward(self, x):
        return self.readout(self.features(x))
