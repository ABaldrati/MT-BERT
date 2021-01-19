from enum import Enum


class Task(Enum):
    CoLA = 1
    SST_2 = 2
    STS_B = 3
    MNLI = 4
    RTE = 5
    WNLI = 6
    QQP = 7
    MRPC = 8
    QNLI = 9
    SNLI = 10
    SciTail = 11

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
