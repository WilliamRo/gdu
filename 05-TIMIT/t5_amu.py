i
import timit_core as core
import timit_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.amu import AMU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'amu'
id = 2
def model(th):
  assert isinstance(th, m.Config)
  cells = []
  for _ in range(th.num_layers):
    cells.append(AMU(
      output_dim=th.num_units,
      neurons_per_amu=th.unit_size,
      truncate_grad=th.truncate_grad,
    ))
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
    
  L1
  2x44: 10189
  3x30: 10225
  4x23: 10444
  L2
  2x29: 10146
  3x20: 9945
  L3
  2x23: 9938
  3x16: 9785
  """
  th.num_layers = 1

  th.unit_size = 3
  th.num_units = 30

  # Setting truncate_grad to False works better
  th.truncate_grad = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0008

  th.validation_per_round = 4
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({}x{})'.format(model_name, th.unit_size, th.num_units)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
