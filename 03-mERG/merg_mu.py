from tframe import Classifier

from tframe.layers import Input, Activation, Rescale
from tframe.models import Recurrent

from tframe.trainers.smartrainer import SmartTrainerHub as Config
from tframe.layers.advanced import Dense


def typical(th, cell):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))
  # Add hidden layers
  model.add(cell)
  # Build model and return
  output_and_build(model, th)
  return model


def output_and_build(model, th):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add output layer
  model.add(Dense(num_neurons=th.output_dim))
  model.add(Activation('softmax'))

  model.build(metric='gen_acc', batch_metric='gen_acc',
              val_targets='val_targets')
