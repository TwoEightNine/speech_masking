import dct_masking
import utils
import wav

# masking config
batch_size = 2 ** 18
batch_offset = 2 ** 13
cut_perc = (1, 99)

amps = wav.read_orig()

print('Perform spectrum changes...')
new_amps = dct_masking.mask(amps, batch_size, batch_offset)

print('Almost done...')
new_amps = utils.cut_peaks(new_amps, cut_perc)

wav.save_result('masked', new_amps)

print('Perform spectrum reverting...')
new_amps2 = dct_masking.unmask(new_amps, batch_size, batch_offset)

print('Almost done...')
new_amps2 = utils.cut_peaks(new_amps2, cut_perc)

wav.save_result('restored', new_amps2)
