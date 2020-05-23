# -*- coding: utf-8 -*-
#%%
from dynaconf import settings
from torch.utils.data import Dataset
import glob
import pathlib, os.path

#%% [markdown]
# ways to access: (<filename>, <startime>)
# - [ ] get random selections perhaps with specification of "normal" or "seizure"
#       dataset = RandomClipDataSet(num_clips=100, length=12)
#       dataset[5] -> 12 second clip at random

# (file_name, start, stop) -> clip

# could write a dataloader to iterate through seizures and non-seizures
# how to do split? 5-fold or leave-out one validation?
# [how to load in parallel](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
# [pytorch custom datasets and loaders](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
#%%
HDF_PATH = pathlib.Path(settings.EEGML_STEV_NEONATES) / 'hdf'

FILE_NAMES = ['eeg1.eeg.h5',
 'eeg2.eeg.h5',
 'eeg3.eeg.h5',
 'eeg4.eeg.h5',
 'eeg5.eeg.h5',
 'eeg6.eeg.h5',
 'eeg7.eeg.h5',
 'eeg8.eeg.h5',
 'eeg9.eeg.h5',
 'eeg10.eeg.h5',
 'eeg11.eeg.h5',
 'eeg12.eeg.h5',
 'eeg13.eeg.h5',
 'eeg14.eeg.h5',
 'eeg15.eeg.h5',
 'eeg16.eeg.h5',
 'eeg17.eeg.h5',
 'eeg18.eeg.h5',
 'eeg19.eeg.h5',
 'eeg20.eeg.h5',
 'eeg21.eeg.h5',
 'eeg22.eeg.h5',
 'eeg23.eeg.h5',
 'eeg24.eeg.h5',
 'eeg25.eeg.h5',
 'eeg26.eeg.h5',
 'eeg27.eeg.h5',
 'eeg28.eeg.h5',
 'eeg29.eeg.h5',
 'eeg30.eeg.h5',
 'eeg31.eeg.h5',
 'eeg32.eeg.h5',
 'eeg33.eeg.h5',
 'eeg34.eeg.h5',
 'eeg35.eeg.h5',
 'eeg36.eeg.h5',
 'eeg37.eeg.h5',
 'eeg38.eeg.h5',
 'eeg39.eeg.h5',
 'eeg40.eeg.h5',
 'eeg41.eeg.h5',
 'eeg42.eeg.h5',
 'eeg43.eeg.h5',
 'eeg44.eeg.h5',
 'eeg45.eeg.h5',
 'eeg46.eeg.h5',
 'eeg47.eeg.h5',
 'eeg48.eeg.h5',
 'eeg49.eeg.h5',
 'eeg50.eeg.h5',
 'eeg51.eeg.h5',
 'eeg52.eeg.h5',
 'eeg53.eeg.h5',
 'eeg54.eeg.h5',
 'eeg55.eeg.h5',
 'eeg56.eeg.h5',
 'eeg57.eeg.h5',
 'eeg58.eeg.h5',
 'eeg59.eeg.h5',
 'eeg60.eeg.h5',
 'eeg61.eeg.h5',
 'eeg62.eeg.h5',
 'eeg63.eeg.h5',
 'eeg64.eeg.h5',
 'eeg65.eeg.h5',
 'eeg66.eeg.h5',
 'eeg67.eeg.h5',
 'eeg68.eeg.h5',
 'eeg69.eeg.h5',
 'eeg70.eeg.h5',
 'eeg71.eeg.h5',
 'eeg72.eeg.h5',
 'eeg73.eeg.h5',
 'eeg74.eeg.h5',
 'eeg75.eeg.h5',
 'eeg76.eeg.h5',
 'eeg77.eeg.h5',
 'eeg78.eeg.h5',
 'eeg79.eeg.h5',
]



class DatasetTemplate(Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def test_numbers():
    dataset = NumbersDataset()
    print(len(dataset))
    print(dataset[100])
    print(dataset[122:361])
