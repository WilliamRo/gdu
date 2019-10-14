import tensorflow as tf
import ap_core as core
import ap_mu as m
from tframe.utils.misc import date_string
from tframe import console
from tframe.nets.rnn_cells.gdu import GDU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gdu'
id = 3

def model(th):
  assert isinstance(th, m.Config)
  cell = GDU(configs=th.gdu_string)
  return m.typical(th, cell)


def main(_):
  console.start('{} on AP task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.sequence_length = 200
  th.terminal_threshold = 0.002

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.gdu_string = '10x10'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.01

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.export_tensors_upon_validation = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_T{}'.format(th.sequence_length)
  th.mark = '{}({})'.format(model_name, th.state_size) + tail
  th.gather_summ_name = th.prefix +  summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()