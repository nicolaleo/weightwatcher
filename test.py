#modifiche di test
#seconda modifica
#terza modifica
#quarta modifica
#quinta modifica
#nnnn
import weightwatcher as ww
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble




#model = models.vgg19_bn(pretrained=True)
model_name='yolov7-e6e.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = attempt_load(model_name, map_location=device)  # load FP32 model

watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(randomize=True, plot=True)

print ("### model details ###")
print("type details:",type(details))
print(details)

from pathlib import Path  
filepath = Path('C:\\MachineLearning_progetti\\MIEI\\weightWather\\'+model_name.split(".")[0]+'.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
details.to_csv(filepath)  


summary = watcher.get_summary(details)
print ("### summary ###")
print(summary)
#details = watcher.analyze(plot=True)#grafici layer per layer
#watcher.describe()
#details = watcher.analyze(randomize=True, plot=True) #grafico con anche Correlation Traps

#description = watcher.describe(model=model)
#print(description)


#model2 = attempt_load('yolov7_massivo_scratch_best.pt', map_location=device)  # load FP32 model
#model_distances = watcher.distances(model, model2)
#print(model_distances)


#cio