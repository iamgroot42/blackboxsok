# Requires installing the robustbench repository from here to use their convenient functions
# https://github.com/RobustBench/robustbench#model-zoo-list-of-models
from robustbench.utils import load_model

model_dir = '/p/blackboxsok/models' # make sure we download all models to the project directory on department server
model_names = ['Standard','Zhang2019Theoretically','Rebuffi2021Fixing_70_16_cutmix_extra'] # for robust models, we can refer to model IDs here: https://github.com/RobustBench/robustbench#model-zoo-list-of-models
dataset = 'cifar10'
for model_name in model_names:
    print("Downloading model:",model_name)
    # we just need the pretrained model weights, not the actual model
    _ = load_model(model_name=model_name,model_dir=model_dir,dataset=dataset)