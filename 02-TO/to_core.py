import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console
from tframe import Predictor
from tframe.trainers.smartrainer import SmartTrainerHub as Config
import to_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Config(as_global=True)
th.data_dir = from_root('02-TO/data')
th.job_dir = from_root('02-TO')
# -----------------------------------------------------------------------------
# Some device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.sequence_length = 100
th.bits = 3

th.input_shape = [6]
th.fixed_length = True
th.val_size = 500

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.batch_size = 20
th.learning_rate = 0.001

th.validate_cycle = 20

th.num_steps = -1
th.val_batch_size = -1
th.print_cycle = 1

th.save_model = False
th.overwrite = True
th.gather_note = True

th.max_iterations = 50000

def activate():
  if not th.fixed_length: th.batch_size = 1
  # This line must be put in activate
  th.output_dim = 2 ** th.bits

  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Predictor)

  # Load data
  train_set, val_set, _ = du.load(
    th.data_dir, th.sequence_length, th.bits, th.fixed_length, th.val_size)

  # Train
  model.train(train_set, validation_set=val_set, trainer_hub=th,
              terminator=lambda metric: metric == 1.0)

  # End
  model.shutdown()
  console.end()
