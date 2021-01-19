import datetime
import operator
from pathlib import Path
from random import sample

import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from torch import optim
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MT_BERT, compute_qnli_batch_output
from task import Task, TaskConfig

NUM_EPOCHS = int(5)

if __name__ == '__main__':
    training_start = datetime.datetime.now().isoformat()
    print(f"------------------ training-start:  {training_start} --------------------------)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    datasets_config = {
        Task.CoLA: TaskConfig(("glue", "cola"), ["label", "sentence"], batch_size=64, metrics=[matthews_corrcoef]),
        Task.SST_2: TaskConfig(("glue", "sst2"), ["label", "sentence"], batch_size=64, metrics=[accuracy_score]),
        Task.STS_B: TaskConfig(("glue", "stsb"), ["label", "sentence1", "sentence2"], batch_size=64,
                               metrics=[pearsonr, spearmanr]),
        Task.MNLI: TaskConfig(("glue", "mnli"), ["label", "hypothesis", "premise"], batch_size=64,
                              metrics=[accuracy_score]),
        Task.WNLI: TaskConfig(("glue", "wnli"), ["label", "sentence1", "sentence2"], batch_size=64,
                              metrics=[accuracy_score]),
        Task.QQP: TaskConfig(("glue", "qqp"), ["label", "question1", "question2"], batch_size=64,
                             metrics=[accuracy_score, f1_score]),
        Task.RTE: TaskConfig(("glue", "rte"), ["label", "sentence1", "sentence2"], batch_size=64, metrics=[accuracy_score]),
        Task.MRPC: TaskConfig(("glue", "mrpc"), ["label", "sentence1", "sentence2"], batch_size=64,
                              metrics=[accuracy_score, f1_score]),
        Task.QNLI: TaskConfig(("glue", "qnli"), ["label", "question", "sentence"], batch_size=64, metrics=[accuracy_score]),
        Task.SNLI: TaskConfig(("snli", "plain_text"), ["label", "hypothesis", "premise"], batch_size=64,
                              metrics=[accuracy_score]),
        Task.SciTail: TaskConfig(("scitail", "tsv_format"), ["label", "hypothesis", "premise"], batch_size=64,
                                 metrics=[accuracy_score])
    }

    tasks_config = {}
    for task, task_config in datasets_config.items():
        dataset_config, columns = task_config.dataset_loading_args, task_config.columns
        train_dataset = load_dataset(*dataset_config, split="train")

        if task == Task.MNLI:
            val_dataset_matched = load_dataset(*dataset_config, split="validation_matched")
            val_dataset_mismatched = load_dataset(*dataset_config, split="validation_mismatched")

            val_dataset_matched.set_format(columns=columns)
            val_dataset_mismatched.set_format(columns=columns)

            val_dataset = concatenate_datasets([val_dataset_matched, val_dataset_mismatched])
        else:
            val_dataset = load_dataset(*dataset_config, split="validation")

        if task == Task.SciTail:
            def label_mapper(x):
                labels = ClassLabel(names=["neutral", "entails"])
                return {"label": labels.str2int(x)}


            train_dataset = train_dataset.map(label_mapper, input_columns=["label"])
            val_dataset = val_dataset.map(label_mapper, input_columns=["label"])

        train_dataset.set_format(columns=columns)
        val_dataset.set_format(columns=columns)

        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=9, batch_size=1, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=9, batch_size=task_config.batch_size,
                                                   shuffle=True)

        tasks_config[task] = {
            "label_feature": train_dataset.features["label"],
            "columns": columns,
            "train_loader": train_loader,
            "val_loader": val_loader
        }

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    results_folder = Path(f"results_{training_start}")
    results_folder.mkdir(exist_ok=True)

    writer = SummaryWriter(str(results_folder / "tensorboard_log"))

    model = MT_BERT()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    task_actions = []
    for task in iter(Task):
        train_loader = tasks_config[task]["train_loader"]
        task_actions.extend([task] * len(train_loader))

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_bar = tqdm(sample(task_actions, len(task_actions)))
        model.train()

        for task_action in epoch_bar:
            train_loader = tasks_config[task_action]["train_loader"]
            epoch_bar.set_description(f"current task: {task_action.name}")

            data = next(iter(train_loader))
            batch_size = data['label'].size(0)
            columns = tasks_config[task_action]["columns"]

            optimizer.zero_grad(set_to_none=True)

            data_columns = [col for col in tasks_config[task_action]["columns"] if col != "label"]
            input_data = list(zip(*(data[col] for col in data_columns)))

            if len(data_columns) == 1:
                input_data = list(map(operator.itemgetter(0), input_data))

            if task_action == Task.QNLI:
                class_label = tasks_config[task_action]["label_feature"]
                output = compute_qnli_batch_output(input_data, class_label, model)
                label = torch.ones(len(output)).to(device)
            else:
                output = model(input_data, task_action)

                label = data["label"]
                if label.dtype == torch.float64:
                    label = label.to(torch.float32)
                label = label.to(device)

            task_criterion = losses[MT_BERT.loss_for_task(task_action)]

            loss = task_criterion(output, label)

            loss.backward()
            optimizer.step()

        models_path = results_folder / "saved_models"
        models_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(models_path / f'epoch_{epoch}.pth'))

        model.eval()
        val_results = {}
        with torch.no_grad():
            for task in iter(Task):
                val_loader = tasks_config[task]["val_loader"]

                task_predicted_labels = torch.empty(0)
                task_labels = torch.empty(0)
                for val_data in val_loader:
                    data_columns = [col for col in tasks_config[task]["columns"] if col != "label"]
                    input_data = list(zip(*(val_data[col] for col in data_columns)))
                    label = val_data["label"]

                    if len(data_columns) == 1:
                        input_data = list(map(operator.itemgetter(0), input_data))

                    model_output = model(input_data, task)
                    if task.num_classes() > 1:
                        predicted_label = torch.argmax(model_output, -1)
                    else:
                        predicted_label = model_output

                    task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                    task_labels = torch.hstack((task_labels, label))

                metrics = datasets_config[task].metrics
                for metric in metrics:
                    val_results[task.name, metric.__name__] = metric(task_labels, task_predicted_labels)
                    writer.add_scalar(f"{task.name}_{metric.__name__}", val_results[task.name, metric.__name__])
        data_frame = pd.DataFrame(
            data=val_results,
            index=[epoch])
        data_frame.to_csv(str(results_folder / f"train_results.csv"), mode='a', index_label='Epoch')
