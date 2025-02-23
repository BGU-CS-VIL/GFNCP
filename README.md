# GFNCP: Consistent Amortized Clustering via Generative Flow Networks [AISTATS 2025]

[Irit Chelly](https://irita42.wixsite.com/mysite), [Roy Uziel](https://uzielroy.wixsite.com/uzielroy), [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/) and [Ari Pakman](https://aripakman.github.io/).

[![arXiv](https://img.shields.io/badge/arXiv-2407.07564-b31b1b.svg?style=flat)](TBD)

<br>
<p align="center">
<img src="https://github.com/BGU-CS-VIL/GFNCP/blob/main/.github/gfncp_fig.png" alt="GFNCP Framework" width="900" height="300">
</p>

Pytorch implementation of GFNCP.

### Requirements
python 3
<br>
torch
<br>
numpy
<br>
wandb

<br><br>

### How to use
Tp run the code:
```
python main.py --dataset MNIST --data_path '/your_path'
```
<br><br>
To use a pretrained model, use the load-model flag, and the latest checkpoint from saved_models folder will be used.
<br><br>
For tracking loss and metrics values during training and evaluation, use wandb flag to log these values to Weights and Biases.
<br><br>
Make sure to update wandb entity and project names in ```main.py```:
```
def init_wandb(args, params):
    if has_wandb:
        wnb = wandb.init(entity='your_entity', project='your_project', name='experiment_name', config=args, settings=wandb.Settings(_service_wait=300))
        ...
```
<br><br>
Use ```params.py``` for setting batch_size, iterations number, and other hyper-parameters.
 
<br><br><br><br>

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


## Citation
If you find this repository helpful, please consider citing our paper:
```
TBD
```
