import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from src.config import cfg
from src.dataset import BirdCLEF_Dataset
from src.model import BirdModel
from src.loss import BCEFocalLoss

def evaluate(model, dataloader, loss_fn):
    model.eval()
    valid_loss = 0
    preds, trues = [], []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()
            preds.append(torch.sigmoid(outputs).cpu())
            trues.append(targets.cpu())
    
    valid_loss /= len(dataloader)
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    
    auc = metrics.roc_auc_score(trues, preds, average='macro', multi_class='ovo')
    f1 = metrics.f1_score(trues, preds > 0.5, average='micro')
    
    return valid_loss, auc, f1

def main():
    # Load validation data
    valid_df = pd.read_csv("path_to_validation.csv")
    valid_dataset = BirdCLEF_Dataset(df=valid_df, augmentation=False, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.VALID_BATCH_SIZE, shuffle=False)
    
    # Load model
    model = BirdModel(
        model_name=cfg.MODEL_NAME,
        pretrained=False,
        in_channels=3,
        num_classes=cfg.NUM_CLASSES,
        pool=cfg.POOL_TYPE
    )
    model.load_state_dict(torch.load("checkpoints/fold_0_best.pth", map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    
    # Define loss function
    loss_fn = BCEFocalLoss(alpha=1).to(cfg.device)
    
    # Evaluate
    valid_loss, auc, f1 = evaluate(model, valid_dataloader, loss_fn)
    print(f"Validation Loss: {valid_loss:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
