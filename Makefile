.PHONY: cnnABEmax cnnminiABEmax cnnBE3 cnnA3A regABEmax regminiABEmax regBE3 regA3A

cnnABEmax: models/CNN/ABE/156B.h5 models/CNN/ABE/157B.h5 models/CNN/ABE/158B.h5

cnnminiABEmax: models/CNN/ABE/243C.h5 models/CNN/ABE/244C.h5

cnnBE3: models/CNN/CBE/89B.h5 models/CNN/CBE/90B.h5

cnnA3A: models/CNN/CBE/160F.h5 models/CNN/CBE/161F.h5

regABEmax: models/regression/ABE/156B.npy models/regression/ABE/157B.npy models/regression/ABE/158B.npy

regminiABEmax: models/regression/ABE/243C.npy models/regression/ABE/244C.npy

regBE3: models/regression/CBE/89B.npy models/regression/CBE/90B.npy

regA3A: models/regression/CBE/160F.npy models/regression/CBE/161F.npy

models/regression/ABE/%.npy:
	python src/regression.py $@
models/regression/CBE/%.npy:
	python src/regression.py $@
models/CNN/ABE/%.h5:
	python src/train.py $@
models/CNN/CBE/%.h5:
	python src/train.py $@
