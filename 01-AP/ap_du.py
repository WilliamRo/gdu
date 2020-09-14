from tframe.data.perpetual_machine import PerpetualMachine
from tframe.data.sequences.benchmarks.ap import AP
from tframe.data.sequences.seq_set import SequenceSet


def load(data_dir, T, fixed_length=True, val_size=500, test_size=500):
  train_set, val_set, test_set = AP.load(
    data_dir, val_size, test_size, T, fixed_length)
  assert isinstance(train_set, PerpetualMachine)
  assert isinstance(val_set, SequenceSet)
  assert isinstance(test_set, SequenceSet)
  return train_set, val_set, test_set



