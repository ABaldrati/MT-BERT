from enum import Enum

import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


class Task(Enum):
    CoLA = 'CoLA'
    SST_2 = 'SST-2'
    STS_B = 'STS-B'
    MNLI = 'MNLI-m'
    RTE = 'RTE'
    WNLI = 'WNLI'
    QQP = 'QQP'
    MRPC = 'MPRC'
    QNLI = 'QNLI'
    SNLI = 'SNLI'
    SciTail = 'SciTail'

    def num_classes(self):
        if self == Task.MNLI or self == Task.SNLI:
            return 3
        elif self == Task.STS_B or self == Task.QNLI:
            return 1
        else:
            return 2


class TaskConfig:
    def __init__(self, dataset_loading_args, columns, batch_size, metrics):
        self.dataset_loading_args = dataset_loading_args
        self.columns = columns
        self.batch_size = batch_size
        self.metrics = metrics


def define_dataset_config():
    datasets_config = {
        Task.CoLA: TaskConfig(("glue", "cola"), ["label", "sentence"], batch_size=32, metrics=[matthews_corrcoef]),
        Task.SST_2: TaskConfig(("glue", "sst2"), ["label", "sentence"], batch_size=32, metrics=[accuracy_score]),
        Task.STS_B: TaskConfig(("glue", "stsb"), ["label", "sentence1", "sentence2"], batch_size=32,
                               metrics=[pearsonr, spearmanr]),
        Task.MNLI: TaskConfig(("glue", "mnli"), ["label", "hypothesis", "premise"], batch_size=32,
                              metrics=[accuracy_score]),
        Task.WNLI: TaskConfig(("glue", "wnli"), ["label", "sentence1", "sentence2"], batch_size=32,
                              metrics=[accuracy_score]),
        Task.QQP: TaskConfig(("glue", "qqp"), ["label", "question1", "question2"], batch_size=32,
                             metrics=[accuracy_score, f1_score]),
        Task.RTE: TaskConfig(("glue", "rte"), ["label", "sentence1", "sentence2"], batch_size=32,
                             metrics=[accuracy_score]),
        Task.MRPC: TaskConfig(("glue", "mrpc"), ["label", "sentence1", "sentence2"], batch_size=32,
                              metrics=[accuracy_score, f1_score]),
        Task.QNLI: TaskConfig(("glue", "qnli"), ["label", "question", "sentence"], batch_size=32,
                              metrics=[accuracy_score]),
        Task.SNLI: TaskConfig(("snli", "plain_text"), ["label", "hypothesis", "premise"], batch_size=32,
                              metrics=[accuracy_score]),
        Task.SciTail: TaskConfig(("scitail", "tsv_format"), ["label", "hypothesis", "premise"], batch_size=32,
                                 metrics=[accuracy_score])
    }
    return datasets_config


def define_tasks_config(datasets_config):
    tasks_config = {}
    for task, task_config in datasets_config.items():
        dataset_config, columns = task_config.dataset_loading_args, task_config.columns
        train_dataset = load_dataset(*dataset_config, split="train")

        if task == Task.MNLI:
            val_dataset_matched = load_dataset(*dataset_config, split="validation_matched")
            val_dataset_mismatched = load_dataset(*dataset_config, split="validation_mismatched")

            test_dataset_matched = load_dataset(*dataset_config, split="test_matched")
            test_dataset_mismatched = load_dataset(*dataset_config, split="test_mismatched")

            val_dataset_matched.set_format(columns=columns)
            val_dataset_mismatched.set_format(columns=columns)

            test_dataset_matched.set_format(columns=columns)
            test_dataset_mismatched.set_format(columns=columns)

            val_dataset = concatenate_datasets([val_dataset_matched, val_dataset_mismatched])
            test_dataset = concatenate_datasets([test_dataset_matched, test_dataset_mismatched])
        else:
            val_dataset = load_dataset(*dataset_config, split="validation")
            test_dataset = load_dataset(*dataset_config, split='test')

        if task == Task.SciTail:
            def label_mapper(x):
                labels = ClassLabel(names=["neutral", "entails"])
                return {"label": labels.str2int(x)}

            train_dataset = train_dataset.map(label_mapper, input_columns=["label"])
            val_dataset = val_dataset.map(label_mapper, input_columns=["label"])
            test_dataset = test_dataset.map(label_mapper, input_columns=["label"])
        elif task == Task.SNLI:
            def label_filter(x):
                return x != -1

            train_dataset = train_dataset.filter(label_filter, input_columns=["label"])
            val_dataset = val_dataset.filter(label_filter, input_columns=["label"])
            test_dataset = test_dataset.filter(label_filter, input_columns=["label"])

        train_dataset.set_format(columns=columns)
        val_dataset.set_format(columns=columns)
        test_dataset.set_format(columns=columns.copy().append('idx'))

        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=task_config.batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4, batch_size=8, shuffle=False)

        tasks_config[task] = {
            "label_feature": train_dataset.features["label"],
            "columns": columns,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "test_dataset": test_dataset
        }
    return tasks_config
