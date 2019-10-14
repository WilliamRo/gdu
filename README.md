Grouped Distributor Units (GDU)
=========
Experiments in paper *[Learning Longer-term Dependencies via Grouped Distributor Unit]( https://arxiv.org/abs/1906.08856 )*.
All experiments are done using [tframe](https://github.com/WilliamRo/tframe), which contains a number of neural network APIs based on ```tensorflow```.

<div style="text-align: left">
  <img src="https://github.com/WilliamRo/gdu/blob/master/figures/gdu.png?raw=true" width="600"/>
</div>

*Figure 1: State transition of a GDU group of size 3, update rate 1.0.*

**Requirements**

tensorflow (>=1.8.0) 

librosa (>=0.6.0)

*[other common packages]*

**How to run**

Change directory into `XX-YYYY` folder (e.g. `04-pMNIST`) and run `tX_ZZZ.py` directly. For example,

```shell
william@alienware:~/gdu/04-pMNIST$ python t4_gdu.py
```

**How to config hyper-parameters**

Hyper-parameters can be configured by modifying `tX_ZZZ.py` file. For example, batch size and learning rate can be specified:

```python
...
def main(_):
    ...
    th.batch_size = 200
    th.learning_rate = 0.002
    ...
...
```

Specifying hyper-parameters via command line arguments is also supported.  In this case the corresponding hyper-parameter specification in `tX_ZZZ.py` module will be ignored. For example

```shell
william@alienware:~/gdu/04-pMNIST$ python t4_gdu.py --batch_size=200 --lr=0.002
```

**Where can I find the implementations of the recurrent models?**

Check this directory `tframe/nets/rnn_cells`.