import datetime
import gc
import hashlib
import math
import operator
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from random import sample
import pandas as pd
import pytorch_warmup as warmup
import scipy
import torch
from torch import optim
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
from tqdm import tqdm

from model import MT_BERT
from task import Task, define_dataset_config, define_tasks_config
from utils import stream_redirect_tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def split_n(chunk_length, sequence):
    if type(sequence) == dict:
        key_splits = {}
        for key, subseq in sequence.items():
            key_splits[key] = split_n(chunk_length, subseq)

        splits_count = len(next(iter(key_splits.values())))
        splits = []

        # Now "transpose" from dict of chunked lists to list of dicts (each with a chunk)
        for i in range(splits_count):
            s = {}
            for key, subseq in key_splits.items():
                s[key] = subseq[i]

            splits.append(s)

        return splits

    else:
        splits = []

        splits_count = math.ceil(len(sequence) / chunk_length)
        for i in range(splits_count):
            splits.append(sequence[i * chunk_length:min(len(sequence), (i + 1) * chunk_length)])

        return splits


def retry_with_batchsize_halving(train_task=None):
    def inner(train_fn):
        @wraps(train_fn)
        def wrapper(*args, **kwargs):
            retry = True
            task = train_task or kwargs.get("task")
            input_data = kwargs["input_data"]
            batch_size = len(input_data)
            label = kwargs.get("label", [0] * batch_size)
            optimizer = kwargs["optimizer"]

            while retry and batch_size > 0:
                microbatches = split_n(batch_size, input_data)
                microlabels = split_n(batch_size, label)

                for microbatch, microlabel in zip(microbatches, microlabels):
                    try:
                        new_kwargs = dict(kwargs, input_data=microbatch, label=microlabel)
                        train_fn(*args, **new_kwargs)
                    except RuntimeError as e:
                        print(f"{e} Error in current task {task} with batch size {batch_size}. Retrying...")
                        batch_size //= 2
                        optimizer.zero_grad(set_to_none=True)
                        break
                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    retry = False

            if retry:
                print(f"Skipping {task} batch... (size: {batch_size})")

        return wrapper

    return inner


@retry_with_batchsize_halving()
def train_minibatch(input_data, task, label, model, task_criterion, **kwargs):
    output = model(input_data, task)
    loss = task_criterion(output, label)
    loss.backward()
    del output


def main():
    all_files = ""
    for file in Path(__file__).parent.resolve().glob('*.py'):
        with open(str(file), 'r', encoding='utf-8') as f:
            all_files += f.read()
    print(hashlib.md5(all_files.encode()).hexdigest())

    NUM_EPOCHS = int(10)
    parser = ArgumentParser()
    parser.add_argument("--from-checkpoint")
    parser.add_argument("--train-epochs", type=int)
    args = parser.parse_args()

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)

    task_actions = []
    for task in iter(Task):
        if task not in [Task.SNLI, Task.SciTail, Task.WNLI]:  # Train only GLUE task
            train_loader = tasks_config[task]["train_loader"]
            task_actions.extend([task] * len(train_loader))
    total_steps = len(task_actions)

    model = MT_BERT()
    model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=5e-5)
    initial_epoch = 1
    training_start = datetime.datetime.now().isoformat()

    if args.from_checkpoint:
        print("Loading from checkpoint")
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        training_start = checkpoint["training_start"]
        warmup_scheduler = None
        lr_scheduler = None
    else:
        print("Starting training from scratch")
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=(total_steps * NUM_EPOCHS) // 10)

    if args.train_epochs:
        NUM_EPOCHS = initial_epoch + args.train_epochs - 1
    print(f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    for epoch in range(initial_epoch, NUM_EPOCHS + 1):
        with stream_redirect_tqdm() as orig_stdout:
            epoch_bar = tqdm(sample(task_actions, len(task_actions)), file=orig_stdout)
            model.train()

            for task_action in epoch_bar:
                train_loader = tasks_config[task_action]["train_loader"]
                epoch_bar.set_description(f"current task: {task_action.name} in epoch:{epoch}")

                data = next(iter(train_loader))

                optimizer.zero_grad(set_to_none=True)

                data_columns = [col for col in tasks_config[task_action]["columns"] if col != "label"]
                input_data = list(zip(*(data[col] for col in data_columns)))

                label = data["label"]
                if label.dtype == torch.float64:
                    label = label.to(torch.float32)
                if task_action == Task.QNLI:
                    label = label.to(torch.float32)

                task_criterion = losses[MT_BERT.loss_for_task(task_action)]

                if len(data_columns) == 1:
                    input_data = list(map(operator.itemgetter(0), input_data))

                label = label.to(device)
                train_minibatch(input_data=input_data, task=task_action, label=label, model=model,
                                task_criterion=task_criterion, optimizer=optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                if warmup_scheduler:
                    lr_scheduler.step()
                    warmup_scheduler.dampen()

            results_folder = Path(f"results_{training_start}")
            results_folder.mkdir(exist_ok=True)

            models_path = results_folder / "saved_models"
            models_path.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_start': training_start
            }, str(models_path / f'epoch_{epoch}.tar'))

            model.eval()
            val_results = {}
            with torch.no_grad():
                task_bar = tqdm([task for task in Task if task not in [Task.SNLI, Task.SciTail, Task.WNLI]],
                                file=orig_stdout)
                for task in task_bar:
                    task_bar.set_description(task.name)
                    val_loader = tasks_config[task]["val_loader"]

                    task_predicted_labels = torch.empty(0, device=device)
                    task_labels = torch.empty(0, device=device)
                    for val_data in val_loader:
                        data_columns = [col for col in tasks_config[task]["columns"] if col != "label"]
                        input_data = list(zip(*(val_data[col] for col in data_columns)))
                        label = val_data["label"].to(device)

                        if len(data_columns) == 1:
                            input_data = list(map(operator.itemgetter(0), input_data))

                        model_output = model(input_data, task)

                        if task == Task.QNLI:
                            predicted_label = torch.round(model_output)
                        elif task.num_classes() > 1:
                            predicted_label = torch.argmax(model_output, -1)
                        else:
                            predicted_label = model_output

                        task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                        task_labels = torch.hstack((task_labels, label))

                    metrics = datasets_config[task].metrics
                    for metric in metrics:
                        metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                        if type(metric_result) == tuple or type(metric_result) == scipy.stats.stats.SpearmanrResult:
                            metric_result = metric_result[0]
                        val_results[task.name, metric.__name__] = metric_result
                        print(
                            f"val_results[{task.name}, {metric.__name__}] = {val_results[task.name, metric.__name__]}")
            data_frame = pd.DataFrame(
                data=val_results,
                index=[epoch])
            data_frame.to_csv(str(results_folder / f"train_GLUE_results.csv"), mode='a', index_label='Epoch')


if __name__ == '__main__':
    main()
