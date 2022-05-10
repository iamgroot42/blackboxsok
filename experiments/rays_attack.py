import torch as ch
from bbeval.attacker.top1_score import RayS
from bbeval.datasets.image.cifar10 import CIFAR10Wrapper
from bbeval.models.pytorch_models import Inceptionv3
from bbeval.config import ModelConfig, DatasetConfig, AttackerConfig


if __name__ == "__main__":
    # Get a pretrained ImageNet model
    model = Inceptionv3(ModelConfig(use_pretrained=True))
    model.cuda()
    # Get data-loader, make sure it works
    ds = CIFAR10Wrapper(DatasetConfig(name="cifar10", type="image"))
    _, _, test_loader = ds.get_loaders(batch_size=32)
    x_sample, y_sample = next(iter(test_loader))
    x_sample, y_sample = x_sample.cuda(), y_sample.cuda()
    # For now, make a random image
    x_sample = ch.rand(32, 3, 299, 299).cuda()
    # Create attacker object
    attacker = RayS(model, AttackerConfig(experiment_name="rays_attack", access_level="all", query_budget=200))
    x_sample_adv, queries_used = attacker.attack(x_sample, y_sample, eps=1.0)
