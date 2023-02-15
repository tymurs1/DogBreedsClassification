#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


import argparse
import logging
import io
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# def test(model, test_loader, epoch, hook):
def test(model, test_loader, epoch):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuracy/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
# if hook:
#     hook.set_mode(smd.modes.EVAL)
# test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    logger.info(
        "Train Epoch: {} ; \nValidation set: Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


# def train(model, train_loader, criterion, optimizer, hook):
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    # if hook:
    #     hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
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


def main(args):
    model=net()
    
    # hook = smd.Hook.create_from_json_file()
    # hook.register_hook(model)
    # hook = get_hook(create_if_not_exists=True)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 
    
    train_dir = os.environ['SM_CHANNEL_TRAIN']
    val_dir = os.environ['SM_CHANNEL_VALID']
    test_dir = os.environ['SM_CHANNEL_TEST']
    
    train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root = val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True)
    
    for epoch in range(1, args.epochs + 1):
        # train(model, train_loader, loss_criterion, optimizer, hook)
        # test(model, val_loader, criterion, hook)
        train(model, train_loader, loss_criterion, optimizer, epoch)
        test(model, val_loader, epoch)
     
    path = "PyTorch_model.pt"
    torch.save(model.state_dict(), path)

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
        default=800,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    
    
    args=parser.parse_args()
    
    main(args)
