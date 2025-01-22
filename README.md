# deca2onnx

Convert [DECA](https://github.com/yfeng95/DECA) model to ONNX for easier usage.

Note: original licence for DECA applies to the DECA code and weights - see their repo. Additional code in this repo is MIT licenced.

## Usage

First clone this repo recursively:

```bash
git clone --recursive https://github.com/nlml/deca2onnx
```

Then from the repo root dir:

```bash
mamba env create -f environment.yml
conda activate deca2onnx
```

Then download FLAME (you first need to register and obtain a username and password) and DECA weights.

```bash
./download_flame.sh <USERNAME> <PASSWORD>
gdown --id 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar
```

Finally, export the ONNX/Pytorch JIT traces of DECA with:

```bash
python export_deca_trace.py
```

You can check the results with
```bash
python test_model.py
```
