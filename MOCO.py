#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import logging
import json
from random import random
from tqdm import tqdm
from functools import reduce, partial
import moco.builder
import numpy as np
import logging
import argparse

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L
from paddle.fluid.optimizer import MomentumOptimizer as Mome
from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

# from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_dir', default='./rtedata', type=str,
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.00002, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--from_pretrained', default='ernie-2.0-large-en', type=str,
                        help='pretrained model directory or tag')
    parser.add_argument('--save-epoch', default=40, type=int)
    parser.add_argument('--save-path', default='LUNA_resnet50_128_imagenet', type=str)
    parser.add_argument('--use_lr_decay', action='store_true',
                        help='if set, learning rate will decay to zero at `max_steps`')
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2490, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--save_dir', type=str, default='model', help='model output directory')
    # options for moco v2
    parser.add_argument('--mlp', default=True,
                        help='use mlp head')
    parser.add_argument('--aug-plus', default=True,
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', default=True,
                        help='use cosine lr schedule')
    parser.add_argument('--arch', default='./model',
                        help='use cosine lr schedule')
    parser.add_argument('--max_seqlen', default=64, type=int,
                        help='max length of a sentence')
    args = parser.parse_args()


    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)
    # tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    feature_column = propeller.data.FeatureColumns([
        propeller.data.TextColumn('seg_a', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab,
                                  tokenizer=tokenizer.tokenize),
        propeller.data.TextColumn('seg_b', unk_id=tokenizer.unk_id, vocab_dict=tokenizer.vocab,
                                  tokenizer=tokenizer.tokenize)
    ])


    def map_fn(seg_a, seg_b):
        seg_a, seg_b = tokenizer.truncate(seg_a, seg_b, seqlen=args.max_seqlen)
        sen_q, seg_q = tokenizer.build_for_ernie(seg_a,[])
        sen_k, seg_k = tokenizer.build_for_ernie(seg_b,[])
        return sen_q, seg_q, sen_k, seg_k


    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True,
                                            repeat=False, use_gz=False) \
        .map(map_fn) \
        .padded_batch(args.batch_size, (0, 0, 0, 0))

    # dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
    #                                .map(map_fn) \
    #                                .padded_batch(args.bsz, (0, 0, 0))

    shapes = ([-1, args.max_seqlen], [-1, args.max_seqlen], [-1, args.max_seqlen], [-1, args.max_seqlen])
    types = ('int64', 'int64', 'int64', 'int64')

    train_ds.data_shapes = shapes
    train_ds.data_types = types
    # dev_ds.data_shapes = shapes
    # dev_ds.data_types = types

    place = F.CUDAPlace(1)
    with FD.guard(place):
        model = moco.builder.MoCo(args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

        # if args.init_checkpoint is not None:
        #     log.info('loading checkpoint from %s' % args.init_checkpoint)
        #     sd, _ = FD.load_dygraph(args.init_checkpoint)
        #     model.set_dict(sd)

        g_clip = F.clip.GradientClipByGlobalNorm(1.0)  # experimental
        l2 = F.regularizer.L2Decay(regularization_coeff=args.weight_decay)
        opt = Mome(args.lr, parameter_list=model.parameters(), momentum=args.momentum, regularization=l2)
        num=(args.moco_k+1)//args.batch_size
        for epoch in range(args.epochs):
            losses = []
            for step, d in enumerate(tqdm(train_ds.start(place), desc='training')):
                sen_q, seg_q, sen_k, seg_k = d
                loss = model(sen_q, seg_q, sen_k, seg_k)
                loss.backward()
                opt.minimize(loss)
                model.clear_gradients()
                losses.append(loss.numpy())
                print('[',epoch,']',step,'/',num,'loss:', np.concatenate(losses).mean()) 
            if epoch % 5 == 0 or epoch == args.epochs- 1:
                F.save_dygraph(model.state_dict(), args.save_dir)

            # save_file = pd.DataFrame(data=losslist)
            # save_file.to_csv('ch.csv', index=False, encoding="utf-8", header=None)






