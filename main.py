import pandas as pd
import glob
from dataset import SegmentationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import visualize_image
from train import train_model, validate_model, save_model
def get_df():
    df = {
    "image":[],
    "mask":[]
    }
    for f in glob.glob('./lgg-mri-segmentation/kaggle_3m/*/*'):
        if 'mask' in f:
            df["mask"].append(f)

    for f in df['mask']:
        df["image"].append(f.replace('_mask', ''))
    
    df = pd.DataFrame(df)
    return df
def process_df(df):
    train_df, val_df = train_test_split(df, test_size=0.2)
    import albumentations as A

    data_transforms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
    ])

    train_dataset = SegmentationDataset(train_df, data_transforms)
    val_dataset = SegmentationDataset(val_df, data_transforms)


    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    return train_dataloader, val_dataloader



if __name__ == "__main__":
    df = get_df()
    train_dataloader, val_dataloader = process_df(df)
    train_model(model, train_dataloader, val_dataloader, epochs=10, device='cuda')
    save_model(model, 'model.pth')
    
