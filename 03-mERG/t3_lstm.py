import tensorflow as tf
import merg_core as core
import merg_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.lstms import LSTM


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'lstm'
id = 1

def model(th):
  assert isinstance(th, m.Config)
  cell = LSTM(state_size=th.state_size)
  return m.typical(th, cell)


def main(_):
  console.start('{} on mERG task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.multiple = 10

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.state_size = 100

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_m{}'.format(th.multiple)
  th.mark = prefix + '{}({})'.format(model_name, th.state_size) + tail
  th.gather_summ_name = prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()