import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageFile, Image

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, device):

    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    logger.info(
        "Test set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, epochs, train_loader, val_loader, criterion, optimizer, device):
    
    loader = {'train': train_loader, 'val': val_loader}
    for epoch in range(epochs):  
        for phase in ['train', 'val']:  
            print(f'Epoch: {epoch}, Phase: {phase}')
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            accumulated_corrects = 0 
            number_of_samples = 0

        
            for batch_idx, (data, target) in enumerate(loader[phase]):
                data = data.to(device)
                target = target.to(device)
        
                output = model(data)
                loss = criterion(output, target)
                
                if phase == 'train':  
                    optimizer.zero_grad()  
                    loss.backward()  
                    optimizer.step()  
                    
                percentages, preds = torch.max(output, dim=1)  
                accumulated_corrects += torch.sum(preds == target).item()
                number_of_samples += len(target)
                
                if phase == "train":
                    if batch_idx % 10 == 1:
                        logger.info(
                            "Phase: {}, Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.6f}".format(
                                phase, 
                                epoch,
                                number_of_samples,
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                                accumulated_corrects/number_of_samples
                            )
                        )
            if phase == 'val':
                logger.info(
                            "Phase: {}, Epoch: {} \t Validation set: Accuracy: {:.6f}".format(
                                phase,
                                epoch,
                                accumulated_corrects/number_of_samples
                            )
                )
                        
            
                
    
def net():
    
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(num_features, 133))
    
    return model


def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=net()
    model = model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dir = os.environ['SM_CHANNEL_TRAIN']
    val_dir = os.environ['SM_CHANNEL_VALID']
    test_dir = os.environ['SM_CHANNEL_TEST']
    
    train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform=training_transform)
    val_dataset = torchvision.datasets.ImageFolder(root = val_dir, transform=testing_transform)
    test_dataset = torchvision.datasets.ImageFolder(root = test_dir, transform=testing_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    
    train(model, args.epochs, train_loader, val_loader, loss_criterion, optimizer, device)
    
    test(model, test_loader, device)
     
    path = "PyTorch_model.pt"
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=800,
        metavar="N",
        help="input batch size for testing (default: 800)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    
    
    args=parser.parse_args()
    
    main(args)
