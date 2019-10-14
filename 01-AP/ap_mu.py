from tframe import Predictor

from tframe.layers import Input, Linear, Activation, Rescale
from tframe.models import Recurrent

from tframe.trainers.smartrainer import SmartTrainerHub as Config


def typical(th, cells):
  assert isinstance(th, Config)
  # Initiate a model
  model = Predictor(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))
  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model, th)
  return model


def output_and_build(model, th):
  assert isinstance(model, Predictor)
  assert isinstance(th, Config)
  # Add output layer
  model.add(Linear(
    output_dim=th.output_dim,
    use_bias=th.bias_out_units,
  ))

  model.build(metric='mse', loss='mse', last_only=True)
