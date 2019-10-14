import tensorflow as tf
import timit_core as core
import timit_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.gru import GRU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gru'
id = 1
def model(th):
  assert isinstance(th, m.Config)
  cells = []
  for _ in range(th.num_layers):
    cells.append(GRU(state_size=th.state_size))
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

  """
  Layers   state_size   #params
    1          48        10153
    2          30        10225
    3          24        10417
  """
  th.num_layers = 1
  th.state_size = 48

  layer2size = {1: 48, 2:30, 3:24, 4:20}
  th.state_size = layer2size[th.num_layers]
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
  th.mark = '{}({})'.format(model_name, th.state_size)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
