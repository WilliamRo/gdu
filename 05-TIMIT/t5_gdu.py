import numpy as np
import tensorflow as tf
import timit_core as core
import timit_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.gdu import GDU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gdu'
id = 4
def model(th):
  assert isinstance(th, m.Config)
  cells = []
  for _ in range(th.num_layers):
    cells.append(GDU(configs=th.gdu_string))
  return m.typical(th, cells)


def main(_):
  console.start('{} on TIMIT task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  th.prefix = '{}_'.format(date_string())
  summ_name = model_name
  th.suffix = ''
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  """For SxN GDU, params # = 2x(14+SxN)x(SxN)+25xSxN+25 = (53+2xSxN)xSxN+25
     Denote SxN as x, # = 2*x^2 + 53*x + 25
                      x = (sqrt(2609+8*#)-53)/4
     5000: 38; 10000: 58.6
     2 layers: 37;  3 layers: 29;  
  """
  th.num_layers = 1
  layer2config = {1: '15x2+5x5+3x1',
                  2: '15x2+5x1+2x1',
                  3: '15x1+12x1+2x1',}
  th.gdu_string = '15x2+7x4'
  th.gdu_string = layer2config[th.num_layers]

  th.state_size = sum([
    np.prod([int(x) for x in g.split('x')]) for g in th.gdu_string.split('+')])
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.003

  th.validation_per_round = 4

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.gdu_string)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
