import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

from dataset import AnimalDataset
from models.resnet import Resnet
from models.vit import ViT

from tqdm import tqdm


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Resnet().to(device) if args.model == 'resnet' else ViT().to(device)
    if args.vit_pretrained and args.model == 'vit':
        load_state_dict = torch.load(args.ckpt_path)
        model_state_dict = model.state_dict()
        updata_state_dict = dict()
        for key, value in load_state_dict.items():
            if key in model_state_dict and value.shape == model_state_dict[key].shape:
                updata_state_dict[key] = value
        model_state_dict.update(updata_state_dict)
        model.load_state_dict(model_state_dict)
        print('load vit ckpt successfully')
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    train_data = AnimalDataset('dataset/train.csv', 'dataset', transformers=train_data_transforms)
    test_data = AnimalDataset('dataset/test.csv', 'dataset')
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    loss_record = []
    all_loss = []
    all_acc = []
    for epoch in range(args.epochs):
        
        model.train()
        for iteration, (data, labels) in enumerate(tqdm(train_loader, desc=f'epoch: {epoch}, training')):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
        
        epoch_loss = sum(loss_record) / len(loss_record)
        all_loss.append(epoch_loss)
        print(epoch_loss)
            
        model.eval()
        with torch.no_grad():
            acc_count = 0
            for data, labels in tqdm(test_loader, desc=f'epoch: {epoch}, test dataset evaluating'):
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                predictions = output.argmax(dim=1)
                acc_count += (predictions == labels).sum()
            accuracy = acc_count.item() / len(test_loader.dataset)
            print('The accuracy of test set is {:.4f}'.format(accuracy))
            all_acc.append(accuracy)
            
        scheduler.step()
    
    record = pd.DataFrame({})
    record['loss'] = all_loss
    record['acc'] = all_acc
    record.to_csv(f'{args.model}.csv', index=False)
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit'])
    parser.add_argument('--vit_pretrained', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default='B_16.pth')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    train(args)

