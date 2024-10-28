import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
import wandb
import os
import gc
from src.config import cfg
from src.dataset import BirdCLEF_Dataset
from src.model import BirdModel
from src.utils import set_random_seed
from sklearn.model_selection import StratifiedKFold
from src.loss import BCEFocalLoss

def initialize():
    model = BirdModel(
        model_name=cfg.MODEL_NAME,
        pretrained=True,
        in_channels=3,
        num_classes=cfg.NUM_CLASSES,
        pool=cfg.POOL_TYPE
    ).to(cfg.device)

    if cfg.OPTIMIZER == 'adan':
        optimizer = Adan(model.parameters(), lr=cfg.LR, betas=(0.02, 0.08, 0.01), weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs=cfg.MAX_EPOCH,
        steps_per_epoch=len(train_dataloader),
        max_lr=cfg.LR,
        div_factor=25,
        final_div_factor=0.4
    )
    
    scaler = amp.GradScaler(enabled=cfg.ENABLE_AMP)
    
    if cfg.LOSS_TYPE == "BCEFocalLoss":
        loss_fn = BCEFocalLoss(alpha=1).to(cfg.device)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss().to(cfg.device)
    
    return model, optimizer, scheduler, scaler, loss_fn

def main():
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    
    if cfg.WANDB:
        wandb.init(project="BirdCLEF_cv_ver2", name="experiment_name", config=vars(cfg))
    
    # Load and preprocess data
    train_csv = pd.read_csv(os.path.join(cfg.DATA_DIR, "train_metadata.csv"))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_csv, train_csv['primary_label'])):
        train_csv.loc[valid_idx, 'fold'] = fold
    
    for fold in cfg.INFERENCE_FOLDS:
        train_df = train_csv[train_csv['fold'] != fold]
        valid_df = train_csv[train_csv['fold'] == fold]
        
        # Initialize datasets and dataloaders
        train_dataset = BirdCLEF_Dataset(df=train_df, augmentation=True, mode='train')
        valid_dataset = BirdCLEF_Dataset(df=valid_df, augmentation=False, mode='valid')
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.VALID_BATCH_SIZE, shuffle=False)
        
        # Initialize model, optimizer, scheduler, scaler, loss function
        model, optimizer, scheduler, scaler, loss_fn = initialize()
        
        best_loss = float('inf')
        best_model = None
        
        for epoch in range(cfg.MAX_EPOCH):
            model.train()
            for batch in train_dataloader:
                data, targets = batch
                data, targets = data.to(cfg.device), targets.to(cfg.device)
                
                optimizer.zero_grad()
                with amp.autocast(cfg.ENABLE_AMP):
                    outputs = model(data)
                    loss = loss_fn(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            # Validation
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_dataloader:
                    data, targets = batch
                    data, targets = data.to(cfg.device), targets.to(cfg.device)
                    outputs = model(data)
                    loss = loss_fn(outputs, targets)
                    valid_loss += loss.item()
            
            valid_loss /= len(valid_dataloader)
            if cfg.WANDB:
                wandb.log({"epoch": epoch, "valid_loss": valid_loss})
            
            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = model.state_dict()
                torch.save(best_model, f'checkpoints/fold_{fold}_best.pth')
        
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
