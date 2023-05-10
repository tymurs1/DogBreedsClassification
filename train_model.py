#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
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


#TODO: Import dependencies for Debugging andd Profiling

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

def test(model, test_loader, device, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    correct = 0
    running_loss = 0
    number_of_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            running_loss += loss.item() * data.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
            number_of_samples += len(target)

            logger.info(
                "Test set: Accuracy: {}/{} ({:.0f}%) test Loss: {:.6f}".format(
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                    running_loss/number_of_samples
                )
            )

def train(model, epochs, train_loader, val_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    loader = {'train': train_loader, 'val': val_loader}
    batch_factor = {'train': 10, 'val': 2}
    for epoch in range(epochs):  
        for phase in ['train', 'val']:  
            print(f'Epoch: {epoch}, Phase: {phase}')
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
                
            accumulated_loss = 0  
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
                accumulated_loss += loss.item() * data.size(0)
                accumulated_corrects += torch.sum(preds == target).item()
                number_of_samples += len(target)
                
                # if phase == "train":
                if batch_idx % batch_factor[phase] == 1:
                    logger.info(
                        "Phase: {}, Epoch: {} [{}/{} ({:.0f}%)] {} Loss: {:.6f}\t Accuracy: {:.6f}".format(
                            phase, 
                            epoch,
                            number_of_samples,
                            len(loader[phase].dataset),
                            100.0 * batch_idx / len(loader[phase]),
                            phase,
                            accumulated_loss/number_of_samples,
                            accumulated_corrects/number_of_samples
                            )
                        )
                        
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(num_features, 133))
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=net()
    model = model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss()
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)
    
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
    model_dir = os.environ['SM_MODEL_DIR']
    
    train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform=training_transform)
    val_dataset = torchvision.datasets.ImageFolder(root = val_dir, transform=testing_transform)
    test_dataset = torchvision.datasets.ImageFolder(root = test_dir, transform=testing_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    
    train(model, args.epochs, train_loader, val_loader, loss_criterion, optimizer, device, hook)
    
    test(model, test_loader, device, loss_criterion, hook)
    
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
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
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
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
