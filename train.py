import datetime
import gc
import hashlib
import operator
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import List, Any

import pandas as pd
import pytorch_warmup as warmup
import scipy
import torch
from torch import optim
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
from tqdm import tqdm

from model import MT_BERT
from task import Task, define_dataset_config, define_tasks_config

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def train_qnli_batch(batch, class_label, model, loss_function):
    questions = batch["question"]
    answers = batch["sentence"]
    labels = batch["label"]
    relevant_answers = defaultdict(list)
    for question, answer, label in zip(questions, answers, labels):
        if class_label.int2str(torch.tensor([label]))[0] == "entailment":
            relevant_answers[question].append(answer)

    for question, answer, label in zip(questions, answers, labels):
        softmax_answers: List[Any] = answers.copy()
        if class_label.int2str(torch.tensor([label]))[0] == "not_entailment":
            continue
        for relevant_answer in relevant_answers[question]:
            softmax_answers.remove(relevant_answer)

        softmax_answers.append(answer)
        model_input = []
        for a in softmax_answers:
            model_input.append([question, a])
        out_of_memory = False
        try:
            model_output = model(model_input, Task.QNLI)
            loss = loss_function(torch.softmax(model_output, -1)[-1].view(-1), torch.ones(1).to(device))
            loss.backward()
            del model_output
        except RuntimeError as e:
            print(f"{e} \n ---------------------------  in task QNLI\n")
            out_of_memory = True
        try:
            if out_of_memory:
                gc.collect()
                torch.cuda.empty_cache()
                model_output = model(model_input[len(model_input) // 2:], Task.QNLI)
                loss = loss_function(torch.softmax(model_output, -1)[-1].view(-1), torch.ones(1).to(device))
                loss.backward()
                del model_output
        except RuntimeError as e:
            gc.collect()
            print(f"{e} \n ---------- Skipping batch\n")
        torch.cuda.empty_cache()


def main():
    all_files = ""
    for file in Path(__file__).parent.resolve().glob('*.py'):
        with open(str(file), 'r', encoding='utf-8') as f:
            all_files += f.read()
    print(hashlib.md5(all_files.encode()).hexdigest())

    NUM_EPOCHS = int(5)
    parser = ArgumentParser()
    parser.add_argument("--from-checkpoint")
    args = parser.parse_args()

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)

    task_actions = []
    for task in iter(Task):
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

    print(f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    task_actions = []
    for task in iter(Task):
        train_loader = tasks_config[task]["train_loader"]
        task_actions.extend([task] * len(train_loader))
    for epoch in range(initial_epoch, NUM_EPOCHS + 1):

        epoch_bar = tqdm(sample(task_actions, len(task_actions)))
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
            label = label.to(device)

            task_criterion = losses[MT_BERT.loss_for_task(task_action)]

            if len(data_columns) == 1:
                input_data = list(map(operator.itemgetter(0), input_data))

            if task_action == Task.QNLI:
                class_label = tasks_config[task_action]["label_feature"]
                train_qnli_batch(data, class_label, model, losses[MT_BERT.loss_for_task(task_action)])
            else:
                out_of_memory = False
                try:
                    output = model(input_data, task_action)
                    loss = task_criterion(output, label)
                    loss.backward()
                    del output
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    out_of_memory = True
                    print(f"{e} \n------------------------------- in current task: {task_action.name}\n")

                try:
                    if out_of_memory:
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        current_batch_size = len(input_data)
                        first_half_input_data = input_data[:current_batch_size // 2]
                        first_half_label = label[:current_batch_size // 2]
                        second_half_input_data = input_data[round(current_batch_size / 2):]
                        second_half_label = label[round(current_batch_size / 2):]

                        output_first_half = model(first_half_input_data, task_action)
                        loss_first_half = task_criterion(output_first_half, first_half_label)
                        loss_first_half.backward()
                        del output_first_half
                        torch.cuda.empty_cache()

                        output_second_half = model(second_half_input_data, task_action)
                        loss_second_half = task_criterion(output_second_half, second_half_label)
                        loss_second_half.backward()
                        del output_second_half
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"{e} \n ---------- Skipping batch\n")

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
            task_bar = tqdm(Task)
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
                        predicted_label = torch.logical_not(predicted_label)
                        predicted_label.to(torch.int8)
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
                    print(f"val_results[{task.name}, {metric.__name__}] = {val_results[task.name, metric.__name__]}")
        data_frame = pd.DataFrame(
            data=val_results,
            index=[epoch])
        data_frame.to_csv(str(results_folder / f"train_results.csv"), mode='a', index_label='Epoch')


if __name__ == '__main__':
    main()
