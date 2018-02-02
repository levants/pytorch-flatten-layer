"""
Created on Nov 24, 2017

Flatten layer test cases

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from flayer.flatten import (Vectorizer, Flatten)


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
        result = F.log_softmax(x, dim=1)

        return result


class TestFlattenAndVectorization(unittest.TestCase):
    """Test cases for directory data loader"""

    def setUp(self):
        self.vec_tensor = Variable(torch.randn(1, 1, 28, 28))
        self.flt_tensor = Variable(torch.randn(1, 32, 7, 7))
        self.lbl_tensor = Variable(torch.randn(1, 10))
        self.vectorizer = Vectorizer()
        self.flatten = Flatten(50)
        self.net = Net()

    def test_vectorization(self):
        """Test "Vectorizer" layer"""

        x = self.vec_tensor
        x = self.vectorizer(x)

        print("Vectorizer - x.size()", x.size())
        self.assertEqual(x.size(1), 784, 'Size does not match after Vectorizer layer')

    def test_flatten(self):
        """Test "Flatten" layer"""

        x = self.flt_tensor
        x = self.flatten(x)

        print("Flatten - x.size()", x.size())
        self.assertEqual(x.size(1), 50, 'Size does not match after Flatten layer')

    def test_network(self):
        """Test "Flatten" layer"""

        x = self.vec_tensor
        x = self.net(x)

        print("Flatten - x.size()", x.size())
        self.assertEqual(x.size(1), 10, 'Size does not match after network model')
