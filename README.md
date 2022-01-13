# Anonymous IJCAI submission
> Paper ID 813

> Note that we only release the codes and pretrained model for our final model, `P2CNet`, trained on mix dataset.

## Requirements
- PyTorch 1.7.0
- Others could be installed directly via pip.

## Dataset
- The datasets derived from SpaceNet could be downloaded from [Google Drive](https://drive.google.com/file/d/18rD0zQJHniBkLzuuwBICm1rG2f8XMI1l/view?usp=sharing).
- The datasets derived from OSM could be downloaded from [Google Drive](https://drive.google.com/file/d/1EcvUxKxjez6t3qsWc1H0FUUACD92zbBm/view?usp=sharing).
- Suppose we are now inside the `Road Completion` folder, do as follows:
```
> cd ..
> mkdir data
# Download the dataset to 'data' and unzip.
```

## Pretrained Models
- `P2CNet` trained on mix SpaceNet and OSM datasets could be downloaded from [Google Drive](https://drive.google.com/file/d/1OaIO8EHuFx3Sk_haFeaZY6Xcr6EajWAs/view?usp=sharing).
- Unzip the files inside `Road Completion` folder.

## Test
- RunL
```python
> python test_deeplabv3plus_mix_mp_sat_gsam_spacenet.py
> python test_deeplabv3plus_mix_mp_sat_gsam_osm.py
```
