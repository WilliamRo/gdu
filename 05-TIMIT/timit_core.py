import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console, SaveMode
from tframe.models.sl.classifier import Classifier
from tframe.trainers import SmartTrainerHub
import timit_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('05-TIMIT/data')
th.job_dir = from_root('05-TIMIT')
# -----------------------------------------------------------------------------
# Some device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.3

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [13]
th.output_dim = 25
th.last_only = True

# -----------------------------------------------------------------------------
# Set common model configs
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 1000
th.num_steps = -1

th.shuffle = True

th.early_stop = True
th.patience = 5

th.batch_size = 1
th.val_batch_size = -1

th.save_model = True
th.save_mode = SaveMode.ON_RECORD

th.overwrite = True
th.gather_note = True
th.export_tensors_upon_validation = True

th.print_cycle = 5

# th.evaluate_train_set = True
th.validate_test_set = True


def activate():
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Load data
  train_set, test_set = du.load(th.data_dir, random=True)

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=train_set, trainer_hub=th,
                test_set=test_set,
                evaluate=lambda t: du.TIMIT25.evaluate(t, test_set))
  else:
    model.evaluate_model(train_set, batch_size=-1)
    model.evaluate_model(test_set, batch_size=-1)

  # End
  model.shutdown()
  console.end()
