.PHONY: modelsABEmax modelsminiABEmax modelsBE3 modelsA3A

modelsABEmax: models/ABE/156B.h5 models/ABE/157B.h5 models/ABE/158B.h5

modelsminiABEmax: models/ABE/243C.h5 models/ABE/244C.h5

modelsBE3: models/CBE/89B.h5 models/CBE/90B.h5

modelsA3A: models/CBE/160F.h5 models/CBE/161F.h5


models/ABE/%.h5:
	python src/train.py $@
models/CBE/%.h5:
	python src/train.py $@
