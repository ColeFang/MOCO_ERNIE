import os
import re
import time
import logging
import json
from random import random
from tqdm import tqdm
from functools import reduce, partial

import numpy as np
import logging
import argparse
import pandas as pd
import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L

from propeller import log
import propeller.paddle as propeller
import math
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay

class MoCo(ErnieModelForSequenceClassification):
    def forward(self, labels, *inputs, **kwargs):
        logits = super(InferemceModel, self).forward(*args, **kwargs)
        if len(labels.shape) == 1:
            labels = L.reshape(labels, [-1, 1])
        loss = L.softmax_with_cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
        return loss, logits



if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument('--from_pretrained', type=str, default='ernie-2.0-large-en', help='pretrained model directory or tag')
    parser.add_argument('--max_seqlen', type=int, default=128, help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=16, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument('--data_dir', type=str, default='./data/CoLA', help='data directory includes train / develop data')
    parser.add_argument('--use_lr_decay', action='store_true', help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='if use_lr_decay is set, '
            'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='./model', help='model output directory')
    parser.add_argument('--max_steps', type=int, default=None, help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')


    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn('seg_a', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab, tokenizer=tokenizer.tokenize),
        propeller.data.LabelColumn('label', vocab_dict={
            b"0": 0,
            b"1": 1,
        }),
    ])

    def map_fn(seg_a, label):
        seg_a = tokenizer.truncate(seg_a, seqlen=args.max_seqlen)
        sentence, segments = tokenizer.build_for_ernie(seg_a)
        return sentence, segments, label

    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0))

    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz, (0, 0))

    test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False,
                                          repeat=False, use_gz=False) \
        .map(map_fn) \
        .padded_batch(args.bsz, (0, 0))

    shapes = ([-1, args.max_seqlen], [-1])
    types = ('int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    dev_ds.data_shapes = shapes
    dev_ds.data_types = types
    test_ds.data_shapes = shapes
    test_ds.data_types = types
    mcc = []


    place = F.CUDAPlace(0)
    with FD.guard(place):
        model = MoCo.from_pretrained('ernie-2.0-large-en', num_labels=128, name='')         
        state_dict = F.load_dygraph('./moco')[0]
        model.load_dict(state_dict)
        fc_features = 128
        model.classifier = FD.Linear(fc_features, 2) 
        if args.init_checkpoint is not None:
            log.info('loading checkpoint from %s' % args.init_checkpoint)
            sd, _ = FD.load_dygraph(args.init_checkpoint)
            model.set_dict(sd)

        g_clip = F.clip.GradientClipByGlobalNorm(1.0) #experimental
        if args.use_lr_decay:
            opt = AdamW(learning_rate=LinearDecay(args.lr, int(args.warmup_proportion * args.max_steps), args.max_steps), parameter_list=model.parameters(), weight_decay=args.wd, grad_clip=g_clip)
        else:
            opt = AdamW(args.lr, parameter_list=model.parameters(), weight_decay=args.wd, grad_clip=g_clip)

        for epoch in range(args.epoch):
            for step, d in enumerate(tqdm(train_ds.start(place), desc='training')):
                ids, sids, label = d
                loss, _ = model(ids, sids, labels=label)
                loss.backward()
                if step % 10 == 0:
                    log.debug('train loss %.5f lr %.3e' % (loss.numpy(), opt.current_step_lr()))
                opt.minimize(loss)
                model.clear_gradients()
            with FD.base._switch_tracer_mode_guard_(is_train=False):
                model.eval()
                FP = 0
                TP = 0
                FN = 0
                TN = 0
                for step, d in enumerate(tqdm(dev_ds.start(place), desc='evaluating %d' % epoch)):
                    ids, sids, label = d
                    loss, logits = model(ids, sids, labels=label)
                    #print('\n'.join(map(str, logits.numpy().tolist())))
                    a = L.argmax(logits, -1).numpy()
                    for i in range(args.bsz):
                        if a[i] == label[i] and a[i] == 1:
                            TP += 1
                        elif a[i] == label[i] and a[i] == 0:
                            TN += 1
                        elif a[i] != label[i] and a[i] == 1:
                            FP += 1
                        elif a[i] != label[i] and a[i] == 0:
                            FN += 1
                mcc.append((TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        if args.save_dir is not None:
            with open(args.save_dir+'dev_perfoemance.txt','r',encoding='utf-8') as f:
                for i in range(len(mcc)):
                    f.write('epoch:'+str(i+1)+'     '+'mcc:'+str(mcc[i]))
                    f.write('\n')
            log.debug('saving inference model')
            test = []
            with FD.base._switch_tracer_mode_guard_(is_train=False):
                model.eval()
                for step, d in enumerate(tqdm(test_ds.start(place), desc='test')):
                    ids, sids, label = d
                    loss, logits = model(ids, sids, labels=label)
                    #print('\n'.join(map(str, logits.numpy().tolist())))
                    a = L.argmax(logits, -1).numpy()
                    test.extend(list(a))
            my_df = pd.DataFrame(test)
            column = ['index', 'prediction']
            my_df.to_csv(args.save_dir+'results.tsv', seq='\t', index=True, header=column)
            class InferemceModel(MoCo):
                def forward(self, *args, **kwargs):
                    _, logits = super(InferemceModel, self).forward(*args, **kwargs)
                    return logits
            model.__class__ = InferemceModel #dynamic change model type, to make sure forward output doesn't contain `None`
            src_placeholder = FD.to_variable(np.ones([1, 1], dtype=np.int64))
            sent_placehodler = FD.to_variable(np.zeros([1, 1], dtype=np.int64))
            model(src_placeholder, sent_placehodler)
            _, static_model = FD.TracedLayer.trace(model, inputs=[src_placeholder, sent_placehodler])
            static_model.save_inference_model(args.save_dir)
            log.debug('done')
