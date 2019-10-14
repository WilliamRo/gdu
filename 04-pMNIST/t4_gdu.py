import pm_core as core
import tensorflow as tf
from tframe import console
from tframe.utils.misc import date_string
import pm_mu as m
from tframe.nets.hyper.gdu_h import GDU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gdu'
id = 3
def model(th):
  assert isinstance(th, m.Config)
  cells = []
  for _ in range(th.num_layers):
    cell = GDU(
      configs=th.gdu_string,
      dropout=th.rec_dropout,
    )
    cells.append(cell)
  return m.typical(th, cells)


def main(_):
  console.start('{} on pMNIST task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.permute = True

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

  th.num_layers = 1
  th.state_size = 256
  th.unit_size = 5
  th.num_units = th.state_size // th.unit_size
  th.delta = 1.0

  th.rec_dropout = 0.4
  th.output_dropout = 0.2

  th.gdu_string = '{}x{}x{}'.format(th.unit_size, th.num_units, th.delta)
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 100
  th.validation_per_round = 10

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.clip_threshold = 1.0
  th.clip_method = 'value'

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_{}'.format('P' if th.permute else 'NP')
  th.mark = '{}({}{})'.format(
    model_name, th.gdu_string, '-r' if th.use_reset_gate else '') + tail

  th.mark += '_bs{}lr{}rd{}od{}gc{}'.format(
    th.batch_size, th.learning_rate, th.rec_dropout, th.output_dropout,
    th.clip_threshold)

  th.gather_summ_name = th.prefix +  summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()



