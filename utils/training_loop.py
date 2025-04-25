import torch
import random
import numpy as np
from chempy.bmin.realtime import model
from torch.backends import cudnn
import torch.optim as optim
from tqdm import tqdm


def training_loop(config):
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    train_set = Dataset(config)
    test_set = Dataset(config)

    trian_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    optimizer = getattr(optim, config["optimizer"])(
        params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    scheduler = getattr(optim.lr_scheduler, config["scheduler"])(
        optimizer,
        T_max=config["scheduler_t_max"],
        eta_min=config["scheduler_eta_min"]
    )

    loss_fn = get_loss_fn(config['loss_fn'])(config)
    mse_loss_fn = torch.nn.MSELoss()

    initial_epoch = 0
    global_step = 0
    preload = config['preload']

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(
            train_loader, desc=f"Processing Epoch {epoch:02d}"
        )

        total_loss = 0.0
        for step, (X, y) in enumerate(batch_iterator):
            X = X.to(config['device'])
            y = y.to(config['device'])

            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            total_loss += loss.item()
            batch_iterator.set_postfix(
                {"loss": f"{loss.item():6.3f}"}
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # logging here??

            scheduler.step()
            avg_loss = total_loss / len(train_loader)



