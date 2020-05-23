# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
plt.ion

# %%
from dynaconf import settings
import numpy as np
import scipy.io

# %%
clin = pd.read_csv(settings.EEGML_STEV_NEONATES+"/clinical_information.csv", index_col='ID')

# %%
print(clin.columns)


# %% [markdown]
# note, "nan" in this table means that "N/A" or not applicable was used for this field
# This could be changed when reading in the file keep_default_na=False

# %%
clin.head() 

# %%
clin.Diagnosis.unique()

# %%
clin['EEG to PMA (weeks)'].unique()

# %%
pma_all = clin['EEG to PMA (weeks)']
pma_all = pma_all.sort_values()
pma_all.hist(figsize=(10,6))
plt.title('age distribution (PMA)')
plt.tight_layout()

# %%

# %%

# %%
# let's see if these nan (probably blank orginally) diag have no seizures - nope not that
clin['num_reviewer_seizure'] = clin['Number of Reviewers Annotating Seizure']

# %%
clin

# %%
clin_nosz = clin[clin.num_reviewer_seizure == 0]
clin_nosz.count()

# %%

# %%

# %%

# %%

# %%

# %%

# %%
annot = scipy.io.loadmat(settings.EEGML_STEV_NEONATES+'/annotations_2017.mat')

# %%
annot

# %%
A = annot['annotat_new']

# %%
A.shape

# %%
A[0][0].shape

# %%
A[0][3].shape

# %%
B = A.squeeze()

# %%
B.shape

# %%
B[0].shape

# %%
B[9].shape  # EEG 10, 3 reviewers marked each second as either seizure or no seizure

# %%
np.where(B[0][0,:] == 1)

# %%
