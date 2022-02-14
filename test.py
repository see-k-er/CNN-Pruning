import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


import torch.nn.utils.prune as prune
from network_params import get_prune_params, print_sparsity

from models import *

def evaluate_performance(path):
  device="cuda" if torch.cuda.is_available() else "cpu"
  criterion = nn.CrossEntropyLoss()
  model = ResNet18()
  model = model.to(device)
  checkpoint = torch.load(path)

  if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark =True
  model.load_state_dict(checkpoint["net"])

  device="cuda" if torch.cuda.is_available() else "cpu"
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          test_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()


  return( str(100. * correct/total))




if __name__ == '__main__':
  
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    print("Accuracy for no pruning model :" + evaluate_performance("./checkpoint/ckpt.pth"))
    print("Accuracy for one-shot pruning 90% sparsity :"+evaluate_performance("./checkpoint/ckpt_prune_one_shot_90.pth"))   
    print("Accuracy for one-shot pruning 75% sparsity :"+evaluate_performance("./checkpoint/ckpt_prune_one_shot_75.pth")   )
    print("Accuracy for one-shot pruning 50% sparsity :"  +evaluate_performance("./checkpoint/ckpt_prune_one_shot_90.pth")  )
    print("Accuracy for iterative pruning 90% sparsity :"+ evaluate_performance("./checkpoint/ckpt_prune_iterative_90.pth")   ) 
    print("Accuracy for iterative pruning 75% sparsity :" +evaluate_performance("./checkpoint/ckpt_prune_iterative_75.pth")   ) 
    print("Accuracy for iterative pruning 50% sparsity :" + evaluate_performance("./checkpoint/ckpt_prune_iterative_50.pth")  )
   