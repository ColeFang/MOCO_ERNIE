# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import optimization
import paddle.fluid as F
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L
def norm(inputs, dim):
    tp = [1,0]
    mm = L.sqrt(L.reduce_sum(L.elementwise_mul(inputs, inputs), dim=-dim))
    h = L.elementwise_div(inputs, mm, axis=tp[dim])
    return h
class MoCo(D.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=300, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = ErnieModelForSequenceClassification.from_pretrained('ernie-2.0-large-en', num_labels=dim)
        self.encoder_k = ErnieModelForSequenceClassification.from_pretrained('ernie-2.0-large-en', num_labels=dim)

        if mlp:
            dim_mlp = 1024
            self.encoder_q.classifier = D.Sequential(D.Linear(dim_mlp, dim_mlp, act='relu'),  self.encoder_q.classifier)
            self.encoder_k.classifier = D.Sequential(D.Linear(dim_mlp, dim_mlp,act='relu'), self.encoder_k.classifier)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k=param_q  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.queue = L.randn([dim, K])
        self.queue = norm(self.queue, dim=0)

        self.queue_ptr = L.zeros([1], dtype='int32')

    @D.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k = param_k * self.m + param_q * (1. - self.m)

    @D.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
       # assert self.K % batch_size == 0  # for simplic
        # replace the keys at ptr (dequeue and enqueue)
        if ptr==0:
            li = [L.transpose(keys, perm=[1, 0]), L.slice(self.queue, axes=[1], starts=[ptr+batch_size], ends=[self.K+100])]
        elif ptr+batch_size == self.K:
            print(ptr)
            print(keys.shape)
            li = [L.slice(self.queue, axes=[1], starts=[0], ends=[ptr]), L.transpose(keys, perm=[1, 0])]
        else:
            li = [L.slice(self.queue, axes=[1], starts=[0], ends=[ptr]), \
              L.transpose(keys, perm=[1, 0]), \
              L.slice(self.queue, axes=[1], starts=[ptr+batch_size], ends=[self.K+100])]
        self.queue = L.concat(li, axis=1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr = ptr

    def forward(self, sen_q, seg_q, sen_k, seg_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(sen_q, seg_q)  # queries: N
        q = norm(q, dim=1)

        # compute key features
        with D.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #sen_k, idx_unshuffle = self._batch_shuffle_ddp(sen_k)

            k = self.encoder_k(sen_k, seg_k)  # keys: NxC
            k = norm(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)


        l_pos=L.unsqueeze(L.reduce_sum(L.elementwise_mul(q, k), dim=1),axes=[-1])
        # negative logits: NxK
        l_neg = L.matmul(q, self.queue.detach())
        # logits: Nx(1+K)
        logits = L.concat([l_pos, l_neg], axis=-1)
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = L.zeros([logits.shape[0]], dtype='int64')
        
        self._dequeue_and_enqueue(k)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)

        return loss


