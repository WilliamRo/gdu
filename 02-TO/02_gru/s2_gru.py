import sys
sys.path.append('../../')

from tframe.utils.script_helper import Helper
from to_core import Config


s = Helper()
s.register_flags(Config)
s.register('job-dir', '.')
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
bits = 3
L = (100, 200)

s.register('max_iterations', 20000)
s.register('bits', bits)
s.register('sequence_length', L)
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
summ_name += '_{}bits_L{}'.format(bits, L)
summ_name += ''
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('state_size', 100)
s.register('lr', 0.001)

s.run(10)

