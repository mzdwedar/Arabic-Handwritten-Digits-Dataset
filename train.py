import copy
import time

from torch.optim import SGD, lr_scheduler

import torch
from torch.nn import CrossEntropyLoss
import torchvision.transforms as T

from utils.dataset import load_data, get_dataloaders, split_data, extract_dataset
from utils.model import model_builder

def train_model(model, dataloaders, dataset_sizes, optimizer, criterion, scheduler, EPOCHS):
  """
  for i in epochs:
    train phase:
      pass the inputs through the net
      compute the loss
      compute gradients
      update weights
    
    val phase:
      passe the inputs through the net
      compute the loss

  returns:
    save the weights of the best model
  """  
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    print('-'*10)

    for phase in ['train', 'test']:
      if(phase =='train'):
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0

      for inputs, labels in dataloaders[phase]:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _ , preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
    
          if(phase =='train'):
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      if (phase == 'train'):
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]   

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      if(phase =='test' and epoch_acc > best_acc):
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    
    print()
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  model.load_state_dict(best_model_wts)
  return model


if __name__ == '__main__':
    
    transform = T.Compose([
    T.ToTensor()
    ])

    X_train, Y_train, X_test, Y_test = load_data()
    dataloaders, dataset_sizes = get_dataloaders(X_train, Y_train, X_test, Y_test, transform)
    
    model = model_builder()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    EPOCHS = 3

    model = train_model(model, dataloaders, dataset_sizes, optimizer, criterion, exp_lr_scheduler, EPOCHS)

    torch.save(model.state_dict(), 'saved_model/AHDD_resnet18.pt')

