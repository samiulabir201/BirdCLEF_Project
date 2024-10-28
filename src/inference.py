import torch
import pandas as pd
import glob
from tqdm import tqdm
import onnx
import onnxruntime as ort
from src.config import cfg
from src.dataset import BirdCLEF_Dataset

def load_models():
    models = {}
    model_paths = glob.glob("checkpoints/*.pth")
    for path in model_paths:
        fold = int(path.split('_')[1])
        model = BirdModel(
            model_name=cfg.MODEL_NAME,
            pretrained=False,
            in_channels=3,
            num_classes=cfg.NUM_CLASSES,
            pool=cfg.POOL_TYPE
        )
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        model.to(cfg.device)
        model.eval()
        models[fold] = model
    return models

def inference(models, test_dataloader):
    predictions = []
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inference"):
            data = data.to(cfg.device)
            preds = []
            for fold, model in models.items():
                output = model(data)
                preds.append(torch.sigmoid(output).cpu())
            mean_preds = torch.mean(torch.stack(preds), dim=0)
            predictions.append(mean_preds)
    return torch.cat(predictions, dim=0)

def main():
    models = load_models()
    test_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "sample_submission.csv"))
    test_dataset = BirdCLEF_Dataset(df=test_df, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    preds = inference(models, test_dataloader)
    
    submission = pd.DataFrame(preds.numpy(), columns=test_df.columns[1:])
    submission.insert(0, 'row_id', test_df['row_id'])
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
