import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


def load_data():
    data = loadmat('./Train + Test Matlab.mat')
    X_train, X_test = data['XTrain'], data['XTest']
    Y_train = pd.read_csv('csvTrainLabel 60k x 1.csv', header = None)
    Y_test = pd.read_csv('csvTestLabel 10k x 1.csv', header=None)
    X_train = X_train.transpose([3, 0, 1, 2])
    X_test = X_test.transpose([3, 0, 1, 2])

    print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')

    return X_train, Y_train, X_test, Y_test

class CustomDataset(Dataset):
  """
  create a custom dataset
  
  Args:
      Images: ndarry of images, shape: (None, 28, 28, 1)
      labels: a Series with the corresponding classes of Images
      transform: transformation of ndarry to tensor

  returns:
      a Dataset Object
  """
  def __init__(self, images, labels, transform=None):
    self.images = images
    self.transform = transform
    self.labels = labels

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels.iloc[idx, 0]

    if(self.transform):
      image = self.transform(image)

    return image, label

def get_dataloaders(X_train, Y_train, X_test, Y_test, transform):
    datasets = {'train': CustomDataset(X_train, Y_train, transform),
            'test': CustomDataset(X_test, Y_test, transform)
               } 

    dataloaders = {'train': DataLoader(datasets['train'], shuffle=True, batch_size=32), 
                'test': DataLoader(datasets['test'], batch_size=32)
                  }

    dataset_sizes = {'train': len(X_train), 
                    'test': len(X_test)
                    }    

    return dataloaders, dataset_sizes