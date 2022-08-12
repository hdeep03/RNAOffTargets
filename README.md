# RNA Off-Targets from CRISPR-Guided Base Editors
Updated: 8/12/22

## Setup
```
.
├── data
├── experiments
│   ├── CNNPredictivePower
│   │   ├── [7.20.22] CNN Performance.ipynb
│   ├── CrossReplicate
│   │   └── [7.26.22] CrossReplicate.ipynb
│   ├── CrossValidation
│   │   ├── [7.27.22] CNN CrossValidation.ipynb
│   │   ├── [7.28.22] Regression CrossValidation.ipynb
│   ├── G-Quadruplex
│   │   ├── ABEmaxBkgd.png
│   │   ├── ABEmaxBkgdLogit.png
│   │   ├── ABEmaxGQuad.png
│   │   ├── ABEmaxGQuadLogit.png
│   │   └── [4.26.22] G-Quadruplex.ipynb
│   ├── ImportanceScoreWindow
│   │   └── [7.20.22] ImpScoreWindow.ipynb
│   ├── InteractionMap
│   │   ├── [8.4.22] InteractionMap.ipynb
│   ├── RegressionPredictivePower
│   │   ├── [7.20.22] RegressionPerformance.ipynb
│   ├── SecureVariants
│   │   ├── [8.3.22] 156BSecureVariant.ipynb
│   │   ├── [8.8.22] miniABE_SecureVariant.ipynb
│   │   └── [8.9.22] BE3-SecureVariant.ipynb
│   └── VariableInputWindow
│       ├── ABEmax
│       │   ├── 156BCNN-1001.h5
│       │   ├── 156BCNN-101.h5
│       │   ├── 156BCNN-201.h5
│       │   ├── 156BCNN-501.h5
│       │   ├── ABEmaxPearsonR.png
│       │   ├── ABEmaxRMSE.png
│       │   └── ABEmaxSpearmanR.png
│       ├── [7.21.22] VariableInpWindow.ipynb
│       └── miniABEmax
│           ├── 243CCNN-1001.h5
│           ├── 243CCNN-101.h5
│           ├── 243CCNN-201.h5
│           ├── 243CCNN-501.h5
│           ├── miniABEmaxPearsonR.png
│           ├── miniABEmaxRMSE.png
│           ├── miniABEmaxSpearmanR.png
│           └── results.csv
├── models
├── scripts
│   └── extend_fasta_seqs.ipynb
├── src
│   ├── dataloader.py
│   ├── regression.py
│   └── train.py
```
To replicate the analysis, download the data from: https://github.com/caleblareau/CRISPR-ABE-RNA-DATA and https://github.com/caleblareau/CRISPR-CBE-RNA-DATA
into the data/ directory. 

One can then use the commands in the makefile to train models. Additionally, the python jupyter notebooks used to conduct additional experiments may be found in the experiments subfolder. Moreover, the extend fasta script in the scripts folder allows one to extend the fasta sequences for analyses that require more information about the surrounding region.

The code assumes a relative file path with this organization.


