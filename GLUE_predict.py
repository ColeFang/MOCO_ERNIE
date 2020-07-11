"""Finetuning on classification task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time
import paddle
import paddle.fluid as fluid
import paddlehub as hub
from GLUE_dataset import GLUE

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--mission", type=str, required=True, help="The Glue Task You Chose")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=32, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name='ernie_v2_eng_large')
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use accuracy as metrics
    # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
    if args.mission == 'CoLA':
        dataset = GLUE("CoLA")
        data = [[d.text_a] for d in dataset.get_predict_examples()]

    elif args.mission == 'SST-2':
        dataset = GLUE("SST-2")
        data = [[d.text_a] for d in dataset.get_predict_examples()]

    elif args.mission == 'MNLI-m':
        dataset = GLUE("MNLI_m")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'MNLI-mm':
        dataset = GLUE("MNLI_mm")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'QQP':
        dataset = GLUE("QQP")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'QNLI':
        dataset = GLUE("QNLI")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]


    elif args.mission == 'STS-B': 
        dataset = GLUE("STS-B")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'MRPC':
        dataset = GLUE("MRPC")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]
    
    elif args.mission == 'RTE':
        dataset = GLUE("RTE")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'WNLI':
        dataset = GLUE("WNLI")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    elif args.mission == 'AX':
        dataset = GLUE("AX")
        data = [[d.text_a, d.text_b] for d in dataset.get_predict_examples()]

    else:
        print('Please use missions in GLUE')   

    if args.mission != 'STS-B':
        # For ernie_tiny, it use sub-word to tokenize chinese sentence
        # If not ernie tiny, sp_model_path and word_dict_path should be set None
        reader = hub.reader.ClassifyReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=args.max_seq_len,
            sp_model_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path())

        # Construct transfer learning network
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of module need
        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            use_data_parallel=args.use_data_parallel,
            use_cuda=args.use_gpu,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=hub.AdamWeightDecayStrategy())

        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config)


        outlist = cls_task.predict(data=data, return_result=True)
        import pandas as pd
        output_df = pd.DataFrame({"index": np.arange(len(outlist)), "prediction": outlist})
        output_df.to_csv(args.mission+'.tsv', sep="\t", index=False)

    else:
        reader = hub.reader.RegressionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

        pooled_output = outputs["pooled_output"]

        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            use_data_parallel=False,
            use_cuda=args.use_gpu,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=hub.AdamWeightDecayStrategy())

        # Define a regression finetune task by PaddleHub's API
        reg_task = hub.RegressionTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            config=config,
        )
        outlist = reg_task.predict(data=data, return_result=True)
        for item in outlist:
            if item > 5:
                item = 5
            if item < 0:
                item = 0
        import pandas as pd
        output_df = pd.DataFrame({"index": np.arange(len(outlist)), "prediction": outlist})
        output_df.to_csv(args.mission+'.tsv', sep="\t", index=False)


