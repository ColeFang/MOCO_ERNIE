import paddle.fluid as F
from ernie.modeling_ernie import ErnieModelForSequenceClassification
import paddle.fluid.dygraph as FD
import numpy as np
from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
place = F.CUDAPlace(0)
with FD.guard(place):
    model = ErnieModelForSequenceClassification.from_pretrained('ernie-2.0-large-en', num_labels=1024, name='q')
    """load MoCo pretrained model"""
    state_dict = F.load_dygraph('./model')[0]
    for each in state_dict.keys():
        print(each)
    for key in list(state_dict.keys()):
        if 'encoder_q' in key:
            print(key[10:])
            new_key = key[10:]
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
    for key in list(state_dict.keys()):
        if key == 'classifier.0.weight':
            new_key = 'classifier.weight'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.0.bias':
            new_key = 'classifier.bias'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if key == 'classifier.2.weight' or key == 'classifier.2.bias':
            del state_dict[key]
    state_dict['classifier.weight'] = state_dict['classifier.weight'][:1024, :]
    state_dict['classifier.bias'] = state_dict['classifier.bias'][:1024]
    model.load_dict(state_dict)
    sen = np.random.random([16, 64]).astype('int64')
    in_sen = to_variable(sen)
    mask = np.random.random([16, 64]).astype('int64')
    in_mask = to_variable(mask)
    out_dygraph, static_layer = TracedLayer.trace(model, inputs=[in_sen, in_mask])
    static_layer.save_inference_model(dirname='./rte50')
