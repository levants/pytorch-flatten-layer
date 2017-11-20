"""
Created on Nov 19, 2017

Implementation of flatten layer

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from torch import nn
import torch
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn.functional as F


class WeghtData(object):
  
  def _set_owner_layer(self, _owner_layer):
    """Sets this weight matrix owner layer
      Args:
        _owner_layer - owner layer of weights
    """
    self._owner_layer = _owner_layer
    
  def copy_(self, src, async=False, broadcast=True):
    """Copies the elements from source into this tensor
      Args:
        src - source tensor
        async - flag to copy asynchronous
        broadcast - flag tu broadcast
    """
    if src is None:
      super(Parameter, self).copy_(src, async=async, broadcast=broadcast)
    else:
      self._owner_layer.calculate_total(src)
      self._owner_layer.weight.data.copy_(src, async=async, broadcast=broadcast)


class WeightParameter(Parameter):
  """Parameter implementation for flatten layer's weights"""
  
  def __new__(cls, data=None, requires_grad=True):
    return super(WeightParameter, cls).__new__(cls, data, requires_grad=requires_grad)
  
  def _set_owner_layer(self, _owner_layer):
    """Sets this weights owner layer
      Args:
        _owner_layer - owner layer of weights
    """
    
    self._weight_data = WeghtData()
    self._weight_data._set_owner_layer(_owner_layer)  

  @property
  def data(self):
    return self._weight_data


class Flatten(nn.Linear):
  """Flatten layer"""
  
  def __init__(self, out_features, in_features=None, bias=True):
    super(nn.Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self._apply_fns = []
    if in_features is None:
      self.weight = None
      weight_parameter = WeightParameter()
      weight_parameter._set_owner_layer(self)
      self.register_parameter('weight', weight_parameter)
    else:
      self.weight = Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
    
  def reset_parameters(self):
    
    if self.weight is not None:
      stdv = 1. / math.sqrt(self.weight.size(1))
      self.weight.data.uniform_(-stdv, stdv)
      if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)

  def _apply(self, fn):
    """Saves passed function for initialized linear layer
      Args:
        fn - functions to apply
      Returns:
        current object instance
    """
    
    if self.weight is None:
      self._apply_fns.append(fn)
    else:
      super(nn.Linear, self)._apply(fn)
      
    return self

  def _apply_postfactum(self):
    """Applies functions from module"""
    
    for fn in self._apply_fns:
      super(nn.Linear, self)._apply(fn)
      
  def _set_input_dim(self, total_dim):
    """Sets total dimension of input
      Args:
        total_dim - total dimension
    """
    
    self.in_features = total_dim
    self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
    self.register_parameter('weight', self.weight)
    self.reset_parameters()
    self._apply_postfactum()   
          
  def set_total_dim(self, total_dim):
    """Sets total dimension of tensor
      Args:
        total_dim - total dimension
    """
    
    if self.weight is None:
      self._set_input_dim(total_dim)
        
  def calculate_total(self, x):
    """Calculates total dimension of tensor
      Args:
        x - tensor to calculate total dimension
    """
    
    if self.weight is None:
      total_dim = 0 if x is None else np.prod(x.size()[1:])
      self.set_total_dim(total_dim)

  def forward(self, input_tensor):
    
    self.calculate_total(input_tensor)
    x = input_tensor.view(input_tensor.size(0), self.in_features)
    linear_result = F.linear(x, self.weight, self.bias) 
    
    return linear_result

