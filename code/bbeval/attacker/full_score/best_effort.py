from email.mime import image
from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from secml_malware.models.c_classifier_end2end_malware import End2EndModel
from secml.array import CArray
from typing import List
import copy


import scurve
from scurve import progress, utils, draw
from PIL import Image, ImageDraw


class _Color:
    def __init__(self, data, block):
        self.data, self.block = data, block
        s = list(set(data))
        s.sort()
        self.symbol_map = {v: i for (i, v) in enumerate(s)}

    def __len__(self):
        return len(self.data)

    def point(self, x):
        if self.block and (self.block[0] <= x < self.block[1]):
            return self.block[2]
        else:
            return self.getPoint(x)


class ColorGradient(_Color):
    def getPoint(self, x):
        c = ord(self.data[x])/255.0
        return [
            int(255*c),
            int(255*c),
            int(255*c)
        ]


class ColorHilbert(_Color):
    def __init__(self, data, block):
        _Color.__init__(self, data, block)
        self.csource = scurve.fromSize("hilbert", 3, 256**3)
        self.step = len(self.csource)/float(len(self.symbol_map))

    def getPoint(self, x):
        c = self.symbol_map[self.data[x]]
        return self.csource.point(int(c*self.step))


class ColorClass(_Color):
    def getPoint(self, x):
        c = self.data[x]
        if c == 0:
            return [0, 0, 0]
        elif c == 255:
            return [255, 255, 255]
        elif chr(c) in string.printable:
            return [55, 126, 184]
        return [228, 26, 28]


class ColorEntropy(_Color):
    def getPoint(self, x):
        e = utils.entropy(self.data, 32, x, len(self.symbol_map))
        # http://www.wolframalpha.com/input/?i=plot+%284%28x-0.5%29-4%28x-0.5%29**2%29**4+from+0.5+to+1

        def curve(v):
            f = (4*v - 4*v**2)**4
            f = max(f, 0)
            return f
        r = curve(e-0.5) if e > 0.5 else 0
        b = e**2
        return [
            int(255*r),
            0,
            int(255*b)
        ]


def drawmap_square(map, size, csource, filename):
    map = scurve.fromSize(map, 2, size**2)
    c = Image.new("RGB", map.dimensions())
    cd = ImageDraw.Draw(c)
    step = len(csource)/float(len(map))
    for i, p in enumerate(map):
        color = csource.point(int(i*step))
        cd.point(tuple(p), fill=tuple(color))
    c.save(filename)


class BestEffort(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):
        image_type = self.params.get('image_type', None)
        if not image_type:
            raise ValueError('image_type must be specified')
        if image_type not in ['ce', 'ch']:
            raise ValueError('image_type must be one of [ce, ch]')
        

        iterations = self.params['iterations']  # 5
        epsilon = self.params['epsilon']  # 1.0
        x_adv_new = []
        results = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):

            block = None
            if image_type == 'ce':
                csource = ColorEntropy(x_adv_i.bytes, block)
            else:
                csource = ColorHilbert(x_adv_i.bytes, block)
            # drawmap_square("hilbert", 256, csource, 'somewhere.png')
            fc=1

            x_adv_i_feature = End2EndModel.bytes_to_numpy(
                x_adv_i.bytes, self.model.model.get_input_max_length(), 256, False
            )
            x_adv_i.feature = x_adv_i_feature
            y_pred, adv_score, adv_ds, f_obj = fgsm.run(
                CArray(x_adv_i.feature), CArray(y_label[i][1].cpu()))
            results.append(adv_score.tondarray()[0][1])
            real_adv_x = fgsm.create_real_sample_from_adv(
                x_orig_i.path, adv_ds.X)
            x_adv_i_new: MalwareDatumWrapper = copy.deepcopy(x_orig_i)
            x_adv_i_new.bytes = real_adv_x
            x_adv_new.append(x_adv_new)

        stop_queries = 1

        self.logger.add_result(
            queries_used=stop_queries,
            result={
                "adv_preds": results
            })

        # TODO- Convert x_adv_new to appropriate batch
        return x_adv_new, stop_queries
