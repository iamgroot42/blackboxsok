# blackboxsok

Codebase for our SoK on black-box attacks.

### Installing the package

1. Install any dependencies 
2. Install the package via `pip install -e .`  from the `code/` folder.

### Setting things up

Make sure you set the following paths:

```bash
export BLACKBOXSOK_DATA_PATH=/path/to/data
export BLACKBOXSOK_MODELS_PATH=/path/to/models
export BLACKBOXSOK_CACHE_PATH=/path/to/cache
```

If you use our repository/codebase for your experiments, please cite our work as:
```bib
@inproceedings{suya2024sok,
  title={Sok: Pitfalls in Evaluating Black-Box Attacks},
  author={Suya, Fnu and Suri, Anshuman and Zhang, Tingwei and Hong, Jingtao and Tian, Yuan and Evans, David},
  booktitle={IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  year={2024}
}
```

### Reference of Attacks

1. Square Attack: https://github.com/max-andr/square-attack 
2. Bayes Attack: https://github.com/satyanshukla/bayes_attack
3. NES Attack: https://github.com/labsix/limited-blackbox-attacks
4. Rays Attack: https://github.com/uclaml/RayS
