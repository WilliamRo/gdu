from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.sequences.benchmarks.pmnist import pMNIST


def load_data(data_dir, permute, permute_mark='alpha', test_directly=False):
  if test_directly:
    train_set, test_set = pMNIST.load(
      data_dir, train_size=60000, val_size=0, test_size=10000,
      permute=permute, permute_mark=permute_mark)
    # train_set, test_set = ERG.load(path, cheat=cheat, multiple=multiple)
    assert isinstance(train_set, SequenceSet)
    assert isinstance(test_set, SequenceSet)
    return train_set, test_set
  else:
    train_set, val_set, test_set = pMNIST.load(
      data_dir, permute=permute, permute_mark=permute_mark)
    # train_set, test_set = ERG.load(path, cheat=cheat, multiple=multiple)
    assert isinstance(train_set, SequenceSet)
    assert isinstance(val_set, SequenceSet)
    assert isinstance(test_set, SequenceSet)
    return train_set, val_set, test_set


if __name__ == '__main__':
  from pm_core import th
  train_set, test_set = load_data(
    th.data_dir, True, test_directly=True)
  print()
