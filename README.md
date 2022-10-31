# Spartan: Differentiable Sparsity via Regularized Transportation

  - [Dependencies](#dependencies)
  - [Library](#library)
    - [`Sparsifier` options](#sparsifier-options)
    - [Supported layers](#supported-layers)
  - [Experiments](#experiments)
  - [Limitations](#limitations)
  - [Citation](#citation)
  - [License](#license)
  - [Contributing](#contributing)

**Spartan** is an algorithm for training neural networks with sparse parameters, i.e., with parameters with a large fraction of entries that are exactly zero. 
Parameter sparsity helps to reduce both the computational cost of inference and the cost of storing and communicating model parameters.

Spartan was developed to train neural networks to a precisely controllable target level of sparsity while improving on the test accuracy of the resulting models relative to existing methods.
This repository accompanies the paper [Spartan: Differentiable Sparsity via Regularized Transportation](https://arxiv.org/abs/2205.14107) by Kai Sheng Tai, Taipeng Tian, and Ser-Nam Lim.

### An overview of Spartan

Spartan learns a _soft mask_ for the parameters of each layer in the model that is to be sparsified.
This soft mask is represented as a Tensor with values in between 0 and 1.
Over the course of training, the degree of sparsity of these masks is increased until the target level
of sparsity is reached.
Additionally, the "sharpness" of the masks is increased so that the masks better approximate a "hard"
binary-valued mask.

This soft masking process is coupled with an optimization scheme called _dual averaging_ (also referred
to as the _straight-through gradient method_).
Concretely, we project the parameters in the forward pass so that they are truly _k_-sparse, but
this projection operation is treated as an identity map in the backward pass.

At the end of training, the masks are merged into the model parameters to yield the final sparse model.

## Dependencies

Use the provided `environment.yml` file to create a conda environment for this project: 

```
$ conda env create -f environment.yml
$ conda activate spartan
```

## Library

`spartan` is a model sparsification library that can be used to train sparse neural network models. Here is a sketch of how the library is used:

```python
import torch
from spartan.sparsifier import Sparsifier, SparsifierConfig

train_loader = ...
model = ...
sparsifier_cfg = SparsifierConfig(sparsity_mode="spartan")
# Note: Sparsifier replaces modules in `model` inplace
sparsifier = Sparsifier(model, sparsifier_cfg, total_iters=len(train_loader))
# We initialize the optimizer after Sparsifier so that the new parameters are captured
optimizer = ...
loss_fn = ...
for x, y in train_loader:
    yhat = model(x)
    loss = loss_fn(yhat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sparsifier.step()
sparsifier.finalize()
```

In words, this snippet does the following:
1. The `SparsifierConfig` class tells the `Sparsifier` that we want to use the Spartan algorithm to sparsify the model.
2. The `Sparsifier` class converts `nn.Module` instances inplace to their counterpart `spartan.modules.MaskedModule` instances. These instances contain the necessary scaffolding for learning a mask for the parameters of each layer.
3. In each iteration of training, `sparsifier.step()` updates the mask of each `MaskedModule` instance.
4. At the end of training, `sparsifier.finalize()` converts each `MaskedModule` instance inplace to a standard `nn.Module` instance with sparse weights.


### `Sparsifier` options

The `Sparsifier` class admits several options for customizing its behavior. The following are some of the more important options:

- `sparsity_mode`: The sparsification algorithm to use
  - `standard`: Performs iterative magnitude pruning
  - `dual_averaging`: Performs sparsification using dual averaging, i.e., [Top-KAST](https://arxiv.org/abs/2106.03517) (without backward sparsity)
  - `spartan` (default): Performs sparsification using the Spartan algorithm
- `cost_frac_target` (default `0.10`): The target fraction of nonzero parameters
- `block_dims` (default `(1, 1)`): The block dimensions for block sparse training
- `module_prefixes` (default `[""]`): The prefixes of the names of the modules to be considered for sparsification (for example, to sparsify only the modules under `layer1.mlp`, set `module_prefixes = ["layer1.mlp"]`)

See the `spartan.sparsifier.SparsifierConfig` class for the full set of options.

### Supported layers

There are currently `MaskedModule` versions of the following layer types:
- `nn.Linear`
- `nn.Conv2d`
- `nn.MultiheadAttention`

## Experiments

### ImageNet-1K experiments

The experiments presented in the NeurIPS 2022 paper can be reproduced using the scripts in `experiments/scripts`.
We have made available scripts for multi-GPU training using either `torchrun` or Slurm.
You will have to modify the parameters in the given scripts to point to the location of the ImageNet dataset on your machine or cluster.

You can specify training options via command line arguments to `experiments/train.py`. See the scripts in `experiments/scripts` for examples.
The available command line arguments reflect the entries of the `experiments.train.TrainConfig` class.

**Examples:**

- Single GPU ResNet-50 training with a custom batch size:
  ```
  python -m experiments.train arch=resnet50 data.batch_size_per_device=256
  ```

- Multi-GPU ResNet-50 training with `torchrun`:
  ```
  sh experiments/scripts/imagenet-resnet50.sh
  ```

- Multi-GPU ViT-B/16 training with Slurm:
  ```
  sbatch experiments/scripts/slurm-imagenet-vitb16.sh
  ```

- Training with an architecture provided by [timm](https://github.com/rwightman/pytorch-image-models):
  ```
  python -m experiments.train model_lib=timm arch=resnet50
  ```

**Loading data with WebDataset**

The training code supports sharded data loading using [WebDataset](https://github.com/webdataset/webdataset).
This method can help improve training speed if data loading incurs high network transfer costs during training.
You can create a sharded dataset using the `webdataset.ShardWriter` class; for more details, see https://webdataset.github.io/webdataset/creating/.

A sharded ImageNet dataset can be specified as a command line option for `experiments/train.py` using the following brace syntax:
```
python -m experiments.train data.train_dataset_path="/path/to/imagenet/train/shard_{00000000..00001281}.tar"
```

## Limitations

In order to realize wall clock inference speedups with the sparse models trained using this library, the models have to be run using an appropriate sparse kernel (e.g., a [Triton block sparse matmul kernel](https://github.com/openai/triton/blob/master/python/triton/ops/blocksparse/matmul.py) for block sparse GPU inference). This repository does not currently contain the necessary utilities for executing sparse models using these kernels.

## Citation

If you find this work useful, please include the following citation:
```
@inproceedings{tai2022spartan,
    title={{Spartan: Differentiable Sparsity via Regularized Transportation}},
    author={Tai, Kai Sheng and Tian, Taipeng and Lim, Ser-Nam},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```

## License

The majority of Spartan is licensed under CC-BY-NC, however portions of the project are available under separate license terms: DeiT is licensed under the Apache 2.0 license, and PyTorch is licensed under the BSD 3-clause license.

## Contributing

We welcome your pull requests. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more information.
