# README #

Flatten layer for PyTorch

# Get it!
=========

```shell
pip install pytorch-flatten-layer
```

# Use it!
=========

```python
from nn.flatten import Flatten

class Net(nn.Module):
  """Network model with flatten layer
   for character recognition"""
  
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.flatten = Flatten(50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
      
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = self.flatten(x)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    result = F.log_softmax(x)
    
    return result
```

For sequencial model

```python
from nn.flatten import Flatten

nn.Sequential(nn.Conv2d(1, 10, kernel_size=5),
              nn.MaxPool2d(2, 2),
              nn.ReLU(),
              nn.Conv2d(10, 20, kernel_size=5),
              nn.MaxPool2d(2, 2),
              nn.ReLU(),
              nn.Dropout2d(),
              Flatten(50),
              nn.Linear(50, 10))   
```