import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console
from tframe.models.sl.classifier import Classifier
# from tframe.trainers import SmartTrainerHub
from tframe.trainers.smartrainer import SmartTrainerHub as Config
from tframe.data.sequences.seq_set import SequenceSet
import merg_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Config(as_global=True)
th.data_dir = from_root('03-mERG/data')
th.job_dir = from_root('03-mERG')
# -----------------------------------------------------------------------------
# Some device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30
# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [7]
th.output_dim = 7
th.multiple = 7
th.train_size = 1000
th.test_size = 256

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 300
th.validation_per_round = 20
th.num_steps = -1
th.probe_per_round = 50
th.val_batch_size = -1
th.shuffle = True
th.print_cycle = 1
th.gather_note = True
th.sample_num = 2

th.save_model = False
th.overwrite = True
th.show_structure_detail = True

def activate():

  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Load data
  train_set, test_set = du.load_data(
    th.data_dir, train_size=th.train_size, test_size=th.test_size,
    multiple=th.multiple)
  assert isinstance(train_set, SequenceSet)

  # Train or evaluate
  model.train(train_set, validation_set=test_set, trainer_hub=th,
              probe=lambda t, **kwargs: du.ERG.probe(test_set, t, **kwargs))

  # End
  model.shutdown()
  console.end()
