from tframe.data.sequences.signals.timit25 import TIMIT25
from tframe.data.sequences.seq_set import SequenceSet



def load(data_dir, random=True):
  train_set, test_set = TIMIT25.load(
    data_dir, num_train_foreach=5, random=random)
  assert isinstance(train_set, SequenceSet)
  assert isinstance(test_set, SequenceSet)
  return train_set, test_set


