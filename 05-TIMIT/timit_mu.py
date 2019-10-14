import tensorflow as tf
from tframe import Classifier

from tframe.layers import Input, Linear, Activation
from tframe.models import Recurrent

from tframe.configs.config_base import Config


def typical(th, cells):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))
  # Add hidden layers
  if not isinstance(cells, (tuple, list)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  _output_and_build(model, th)
  return model


def _output_and_build(model, th):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)

  if th.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(th.learning_rate)
  elif th.optimizer == 'nesterov':
    th.momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(
      th.learning_rate, th.momentum, use_nesterov=True)
  elif th.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(th.learning_rate)
  else: raise ValueError('!! In this task, th.optimizer must be a string')

  # Add output layer
  model.add(Linear(output_dim=th.output_dim))
  model.add(Activation('softmax'))

  model.build(optimizer=th.get_optimizer(optimizer),
              metric=['loss', 'seq_acc'], batch_metric='seq_acc',
              eval_metric='seq_acc', last_only=True)
