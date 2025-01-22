# deca2onnx

Convert [DECA](https://github.com/yfeng95/DECA) model to ONNX for easier usage.

Note: original licence for DECA applies to the DECA code and weights - see their repo. Additional code in this repo is MIT licenced.

From this directory:

```
mamba env create -f environment.yml
conda activate deca2onnx
```


```bash
./download_flame.sh <USERNAME> <PASSWORD>
gdown --id 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O data/deca_model.tar
```

```
python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
```