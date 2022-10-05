import torch
from data.dataset import get_iter
import numpy as np
import random
from utils.model_helper import IPM_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def rwt_regression_loss(w, y, y_pre):
    y_pre, w = y_pre.to('cpu'), w.to('cpu')

    return ((y_pre.squeeze() - y.squeeze())**2 * w.squeeze()).mean()

def train(model, data, args):
    model.train()
    epochs = args.n_epochs
    optimizer = torch.optim.Adam(
        [
            {'params' : model.rwt.parameters(), 'weight_decay' : 0},
            {'params' : model.hidden_features.parameters()},
            {'params' : model.out.parameters(), 'weight_decay' : 0},
        ],
        lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, 
        amsgrad=False
        )

    dataloader = get_iter(data, args.batch_size)
    for epoch in range(epochs):
        total_loss = []
        mmds = []
        for (x, t, y) in dataloader:
            if args.scale:
                y = args.scaler.transform(y.reshape(-1, 1))
                y = torch.from_numpy(y)
                
            x, t = x.to(args.device), t.to(args.device)
            optimizer.zero_grad()
            y_pre, w, _ = model(x, t)
            loss = rwt_regression_loss(w, y, y_pre)
            
            total_loss.append(loss.data)
            
            mmd = IPM_loss(x, t, w, k=5)
            mmds.append(mmd.data)
            loss = loss + mmd
            
            loss.backward()
            optimizer.step()

        total_loss = np.mean(total_loss)
        
        yield epoch + 1, model, total_loss
