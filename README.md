# HySUPP

An Open-Source Hyperspectral Unmixing Python Package

---

## Data

### Data format

Datasets consists in a dedicated `.mat` file containing the following keys:

* `Y`: original hyperspectral image (dimension `L` x `N`)
* `E`: ground truth endmembers (dimension `L` x `p`)
* `A`: ground truth abundances (dimension `p` x `N`)
* `H`: HSI number of rows
* `W`: HSI number of columns
* `p`: number of endmembers
* `L`: number of channels
* `N`: number of pixels (`N` == `H`*`W`)

For sparse unmixing, a dictionary `D` containing `M` atoms is required.

* `D`: endmembers library (dimension `L` x `M`)
* `M`: number of atoms

We provide a utility script to turn an existing datasets composed of separated files to fit the required format used throughout the toolbox.


### Simulated

We currently provide two simulated datasets that correspond to 2 different scenarios using 6 endmembers:

1. `SimPurePixels` exhibits some pure pixels.
2. `SimHighlyMixed` exhibits pixels having at least 3 mixed materials (up to 6).

### Real

---

## Experiments

Here are a list of the commands used to launch all the experiments for simulated datasets:


### Supervised

#### CPU only

* FCLSU for all noise levels with every endmembers extractor for both simulated datasets:

```shell
python main.py mode=supervised noise.SNR=20,30,40 extractor=VCA,SiVM,SISAL data=SimPurePixels,SimHighlyMixed model=FCLSU runs=10 --multirun
```

#### GPU required

* UnDIP for all noise levels with every endmembers extractor for both simulated datasets:

```shell
python main.py mode=supervised noise.SNR=20,30,40 extractor=VCA,SiVM,SISAL data=SimPurePixels,SimHighlyMixed model=UnDIP runs=10 --multirun
```

### Sparse

#### CPU only

* SUnSAL for 20dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=20 data=SimPurePixels,SimHighlyMixed model=SUnSAL model.lambd=0.7 runs=10 --multirun
```

* SUnSAL for 30dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=30 data=SimPurePixels,SimHighlyMixed model=SUnSAL model.lambd=0.1 runs=10 --multirun
```

* SUnSAL for 40dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=SUnSAL model.lambd=0.01 runs=10 --multirun
```

* CLSUnSAL for 20dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=20 data=SimPurePixels,SimHighlyMixed model=CLSUnSAL model.lambd=0.1 runs=10 --multirun
```

* CLSUnSAL for 30dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=30 data=SimPurePixels,SimHighlyMixed model=CLSUnSAL model.lambd=0.05 runs=10 --multirun
```

* CLSUnSAL for 40dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=CLSUnSAL model.lambd=0.01 runs=10 --multirun
```

* S2WSU for 20 and 30dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=20,30 data=SimPurePixels,SimHighlyMixed model=S2WSU model.lambd=0.01 runs=10 --multirun
```

* S2WSU for 40dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=S2WSU model.lambd=0.001 runs=10 --multirun
```

* SUnAA for SNR=20,30,40dB for both datasets (only 3 runs per configuration):

```shell
python main.py mode=sparse noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=SUnAA runs=3 --multirun
```

##### MATLAB engine required

* MUA_BPT for 20dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=20 data=SimPurePixels,SimHighlyMixed model=MUA_BPT model.sideBPT=14 runs=10 --multirun
```

* MUA_BPT for 30dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=30 data=SimPurePixels,SimHighlyMixed model=MUA_BPT model.sideBPT=10 runs=10 --multirun
```

* MUA_BPT for 40dB SNR for both datasets:

```shell
python main.py mode=sparse noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=MUA_BPT model.sideBPT=6 runs=10 --multirun
```

#### GPU required

* SUnCNN for SNR=20dB for both datasets:

```shell
python main.py mode=sparse noise.SNR=20 data=SimPurePixels,SimHighlyMixed model=SUnCNN model.niters=4000 runs=10 projection=True --multirun
```

* SUnCNN for SNR=30dB for both datasets:

```shell
python main.py mode=sparse noise.SNR=30 data=SimPurePixels,SimHighlyMixed model=SUnCNN model.niters=8000 runs=10 projection=True --multirun
```

* SUnCNN for SNR=40dB for both datasets:

```shell
python main.py mode=sparse noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=SUnCNN model.niters=16000 runs=10 projection=True --multirun
```

### Blind

#### CPU only

* MVCNMF for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=MVCNMF runs=10 --multirun
```

##### MATLAB engine required

* NMFQMV for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=NMFQMV runs=10 --multirun
```


#### GPU required

* CNNAEU for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=CNNAEU runs=10 --multirun
```

* MSNet for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=MSNet runs=10 --multirun
```

* PGMSU for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=PGMSU runs=10 --multirun
```

* EDAA for SNR=20,30,40dB for both datasets:

```shell
python main.py mode=blind noise.SNR=20,30,40 data=SimPurePixels,SimHighlyMixed model=EDAA runs=10 --multirun
```

* MiSiCNet for 20dB for both datasets

```shell
python main.py mode=blind noise.SNR=20 data=SimPurePixels,SimHighlyMixed model=MiSiCNet model.niters=4000 runs=10 projection=True --multirun
```

* MiSiCNet for 30dB for both datasets

```shell
python main.py mode=blind noise.SNR=30 data=SimPurePixels,SimHighlyMixed model=MiSiCNet model.niters=8000 runs=10 projection=True --multirun
```

* MiSiCNet for 40dB for both datasets

```shell
python main.py mode=blind noise.SNR=40 data=SimPurePixels,SimHighlyMixed model=MiSiCNet model.niters=16000 runs=10 projection=True --multirun
```





