from tframe.data.dataset import DataSet
from tframe.data.sequences.reber import ERG


def load_data(path, train_size, test_size, multiple, rule='lstm97'):
  train_set, test_set = ERG.load(
    path, train_size=train_size, test_size=test_size, cheat=False,
    rule=rule, multiple=multiple)
  assert isinstance(train_set, DataSet)
  assert isinstance(test_set, DataSet)
  return train_set, test_set


def histogram(data_set, m):
  import matplotlib.pyplot as plt
  import matplotlib
  from matplotlib import colors
  from matplotlib.ticker import FuncFormatter
  plt.yticks()
  assert isinstance(data_set, DataSet)
  x = data_set.structure
  plt.hist(x, bins=40, facecolor='#cccccc')
  plt.title('Sequence length distribution, m = {}'.format(m))
  plt.xlabel('Length')
  plt.ylabel('Density')
  ax = plt.gca()
  # Set y axis as percent
  def to_percent(y, _):
    usetex = matplotlib.rcParams['text.usetex']
    pct = y * 100.0 / data_set.size
    return '{:.0f}{}'.format(pct, r'$\%$' if usetex else '%')
  ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

  # Show plot
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
  from merg_core import th
  m = 40
  train_set, test_set = load_data(
    th.data_dir, train_size=1000, test_size=256, multiple=m)
  histogram(train_set, m)
  print()
