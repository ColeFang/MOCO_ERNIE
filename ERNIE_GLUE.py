"""Finetuning on classification task """

import argparse
import ast
import paddlehub as hub
from GLUE_dataset import GLUE

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--mission", type=str, default='WNLI', help="The Glue Task You Chose")
parser.add_argument("--num_epoch", type=int, default=5, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    if args.mission == 'CoLA':
        dataset = GLUE("CoLA")
        metrics_choices = ["matthews"]

    elif args.mission == 'SST-2':
        dataset = GLUE("SST-2")
        metrics_choices = ["acc"]

    elif args.mission == 'MNLI-m':
        dataset = GLUE("MNLI_m")
        metrics_choices = ["acc"]

    elif args.mission == 'MNLI-mm':
        dataset = GLUE("MNLI_mm")
        metrics_choices = ["acc"]

    elif args.mission == 'QQP':
        dataset = GLUE("QQP")
        metrics_choices = ["f1","acc"]

    elif args.mission == 'QNLI':
        dataset = GLUE("QNLI")
        metrics_choices = ["acc"]

    elif args.mission == 'STS-B': #那这儿没得Pearson_Spearman Corr
        dataset = GLUE("STS-B")

    elif args.mission == 'MRPC':
        dataset = GLUE("MRPC")
        metrics_choices = ["f1","acc"]
        args.weight_decay = 0.01

    elif args.mission == 'RTE':
        dataset = GLUE("RTE")
        metrics_choices = ["acc"]
        args.learning_rate = 3e-5

    elif args.mission == 'WNLI':
        dataset = GLUE("WNLI")
        metrics_choices = ["acc"]

    else:
        print('Please use missions in GLUE')   

    #改名加前缀
    import sys
    import os
    import shutil
    if args.checkpoint_dir != None:
        for roots, dir, files in os.walk(args.checkpoint_dir + '/params'):
            for file_name in files:
                os.rename(os.path.join(roots, file_name),os.path.join(roots, '@HUB_ernie_v2_eng_large@'+file_name))
    #移动文件
    if args.checkpoint_dir != None:
        for roots, dir, files in os.walk(args.checkpoint_dir + '/params'):
            for file_name in files:
                shutil.move(os.path.join(roots, file_name), args.checkpoint_dir + '/step_40/' + file_name)

    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name='ernie_v2_eng_large')
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)


    if args.mission != 'STS-B': #这个是回归任务，单独拿出来搞


        # Download dataset and use accuracy as metrics
        # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
        # metric should be acc, f1 or matthews
        

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

        # Select finetune strategy, setup config and finetune
        strategy = hub.AdamWeightDecayStrategy(
            warmup_proportion=args.warmup_proportion,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate)

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            use_data_parallel=args.use_data_parallel,
            use_cuda=args.use_gpu,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=strategy)

        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=metrics_choices)

        # Finetune and evaluate by PaddleHub's API
        # will finish training, evaluation, testing, save model automatically
        cls_task.finetune_and_eval()

    else:

        # Download dataset and use RegressionReader to read dataset
        dataset = GLUE("STS-B")
        reader = hub.reader.RegressionReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=args.max_seq_len)

        # Construct transfer learning network
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of ERNIE's module need
        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        # Select finetune strategy, setup config and finetune
        strategy = hub.AdamWeightDecayStrategy(
            warmup_proportion=args.warmup_proportion,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate)

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            eval_interval=300,
            use_data_parallel=args.use_data_parallel,
            use_cuda=args.use_gpu,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=strategy)

        # Define a regression finetune task by PaddleHub's API
        reg_task = hub.RegressionTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            config=config)

        # Finetune and evaluate by PaddleHub's API
        # will finish training, evaluation, testing, save model automatically
        reg_task.finetune_and_eval()
