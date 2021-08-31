import numpy as np

import dct_masking
import masking_rules
import utils
import wav

# masking config
batch_size = 2 ** 12
cut_perc = (1, 99)
masking_rule = masking_rules.RandomOffsetMaskingRule(2 ** 5, 2 ** 7, 10)

amps = wav.read_orig()

print('Perform spectrum changes...')
new_amps = dct_masking.mask(amps, batch_size, masking_rule)

print('Almost done...')
new_amps = utils.cut_peaks(new_amps, cut_perc)
new_amps = utils.smoothen_batch_borders(new_amps, batch_size)

wav.save_result('masked', new_amps)

print('Perform spectrum reverting...')
new_amps2 = dct_masking.unmask(new_amps, batch_size, masking_rule)

print('Almost done...')
new_amps2 = utils.cut_peaks(new_amps2, cut_perc)

wav.save_result('restored', new_amps2)
