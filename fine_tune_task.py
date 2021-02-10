import datetime
import hashlib
import operator
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pytorch_warmup as warmup
import scipy
import torch
from torch import optim
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
from tqdm import tqdm

from model import MT_BERT
from task import Task, define_dataset_config, define_tasks_config
from train_glue import train_minibatch
from utils import stream_redirect_tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def main():
    all_files = ""
    for file in Path(__file__).parent.resolve().glob('*.py'):
        with open(str(file), 'r', encoding='utf-8') as f:
            all_files += f.read()
    print(hashlib.md5(all_files.encode()).hexdigest())

    NUM_EPOCHS = int(10)
    parser = ArgumentParser()
    parser.add_argument("--from-checkpoint")
    parser.add_argument("--fine-tune-task", type=Task, choices=list(Task), required=True)
    parser.add_argument("--dataset-percentage", type=float, default=100)

    args = parser.parse_args()

    model = MT_BERT()
    model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=5e-5)
    initial_epoch = 1
    training_start = datetime.datetime.now().isoformat()

    fine_tune_task = args.fine_tune_task
    dataset_percentage = args.dataset_percentage

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config, dataset_percentage=dataset_percentage)

    epoch_steps = len(tasks_config[fine_tune_task]['train_loader'])
    if args.from_checkpoint:
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_start = checkpoint["training_start"]
        warmup_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=(epoch_steps * NUM_EPOCHS) // 10)

    print(f"Task fine tune of {fine_tune_task.name} with percentage:{dataset_percentage}")

    print(f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    data_columns = [col for col in tasks_config[fine_tune_task]["columns"] if col != "label"]
    task_criterion = losses[MT_BERT.loss_for_task(fine_tune_task)]

    for epoch in range(initial_epoch, NUM_EPOCHS + 1):
        with stream_redirect_tqdm() as orig_stdout:
            epoch_bar = tqdm(tasks_config[fine_tune_task]['train_loader'], file=orig_stdout)
            model.train()

            for data in epoch_bar:
                optimizer.zero_grad(set_to_none=True)
                input_data = list(zip(*(data[col] for col in data_columns)))

                label = data["label"]
                if label.dtype == torch.float64:
                    label = label.to(torch.float32)

                if len(data_columns) == 1:
                    input_data = list(map(operator.itemgetter(0), input_data))

                label = label.to(device)

                train_minibatch(input_data=input_data, task=fine_tune_task, label=label, model=model,
                                task_criterion=task_criterion, optimizer=optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                if warmup_scheduler:
                    lr_scheduler.step()
                    warmup_scheduler.dampen()
            if args.from_checkpoint:
                results_folder = Path(f"results_{training_start}")
            else:
                results_folder = Path(f"results_ST_{training_start}_{fine_tune_task}")
            results_folder.mkdir(exist_ok=True)

            models_path = results_folder / f"saved_model_fine_tuned_{fine_tune_task}, percentage:{dataset_percentage}%"
            models_path.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_start': training_start
            }, str(models_path / f'epoch_{epoch}.tar'))

            model.eval()
            val_results = {}
            evaluate_task(data_columns, datasets_config, fine_tune_task, model, orig_stdout, tasks_config, val_results)
            if fine_tune_task == Task.MNLIm:
                evaluate_task(data_columns, datasets_config, Task.MNLImm, model, orig_stdout, tasks_config,
                              val_results)
            data_frame = pd.DataFrame(
                data=val_results,
                index=[epoch])
            data_frame.to_csv(
                str(results_folder / f"fine_tune_task: {fine_tune_task}, percentage:{dataset_percentage}%.csv"),
                mode='a', index_label='Epoch')


def evaluate_task(data_columns, datasets_config, fine_tune_task, model, orig_stdout, tasks_config, val_results):
    with torch.no_grad():
        val_bar = tqdm(tasks_config[fine_tune_task]['val_loader'], file=orig_stdout, position=0, leave=True)
        task_predicted_labels = torch.empty(0, device=device)
        task_labels = torch.empty(0, device=device)
        for val_data in val_bar:
            val_bar.set_description(fine_tune_task.name)

            input_data = list(zip(*(val_data[col] for col in data_columns)))
            label = val_data["label"].to(device)

            if len(data_columns) == 1:
                input_data = list(map(operator.itemgetter(0), input_data))

            model_output = model(input_data, fine_tune_task)

            if fine_tune_task.num_classes() > 1 or fine_tune_task == Task.QNLI:
                predicted_label = torch.argmax(model_output, -1)
            else:
                predicted_label = model_output

            task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
            task_labels = torch.hstack((task_labels, label))

        metrics = datasets_config[fine_tune_task].metrics
        for metric in metrics:
            metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
            if type(metric_result) == tuple or type(metric_result) == scipy.stats.stats.SpearmanrResult:
                metric_result = metric_result[0]
            val_results[fine_tune_task.name, metric.__name__] = metric_result
            print(
                f"val_results[{fine_tune_task.name}, {metric.__name__}] = {val_results[fine_tune_task.name, metric.__name__]}")


if __name__ == '__main__':
    main()