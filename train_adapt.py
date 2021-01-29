import hashlib
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
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

    NUM_EPOCHS = int(5)
    parser = ArgumentParser()
    parser.add_argument("--from-checkpoint", required=True)
    parser.add_argument("--adaptation-task", type=Task, choices=[Task.SNLI, Task.SciTail], required=True)
    parser.add_argument("--dataset-percentage", type=float, required=True)

    args = parser.parse_args()

    model = MT_BERT()
    model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=5e-5)
    initial_epoch = 1

    checkpoint = torch.load(args.from_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #TODO: check me
    training_start = checkpoint["training_start"]
    adaptation_task = args.adaptation_task
    dataset_percentage = args.dataset_fraction
    print(f"Task Adaptation of {adaptation_task.name} with fraction:{dataset_percentage}")

    print(f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config, dataset_percentage=dataset_percentage)

    data_columns = [col for col in tasks_config[adaptation_task]["columns"] if col != "label"]
    task_criterion = losses[MT_BERT.loss_for_task(adaptation_task)]

    for epoch in range(initial_epoch, NUM_EPOCHS + 1):
        with stream_redirect_tqdm() as orig_stdout:
            epoch_bar = tqdm(tasks_config[adaptation_task]['train_loader'], file=orig_stdout)
            model.train()

            for data in epoch_bar:
                optimizer.zero_grad(set_to_none=True)
                input_data = list(zip(*(data[col] for col in data_columns)))

                train_minibatch(input_data=input_data, task=adaptation_task, label=label, model=model,
                                task_criterion=task_criterion, optimizer=optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            results_folder = Path(f"results_{training_start}")
            results_folder.mkdir(exist_ok=True)

            models_path = results_folder / f"saved_model_adaptation_{adaptation_task}, fraction:{dataset_percentage}%"
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
                val_bar = tqdm(tasks_config[adaptation_task]['val_loader'], file=orig_stdout)
                task_predicted_labels = torch.empty(0, device=device)
                task_labels = torch.empty(0, device=device)
                for val_data in val_bar:
                    val_bar.set_description(adaptation_task.name)

                    input_data = list(zip(*(val_data[col] for col in data_columns)))
                    label = val_data["label"].to(device)

                    model_output = model(input_data, adaptation_task)

                    predicted_label = torch.argmax(model_output, -1)

                    task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                    task_labels = torch.hstack((task_labels, label))

                    metrics = datasets_config[adaptation_task].metrics
                    for metric in metrics:
                        metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                        val_results[adaptation_task.name, metric.__name__] = metric_result
                        print(
                            f"val_results[{adaptation_task.name}, {metric.__name__}] = {val_results[adaptation_task.name, metric.__name__]}")
            data_frame = pd.DataFrame(
                data=val_results,
                index=[epoch])
            data_frame.to_csv(
                str(results_folder / f"adaptation_task: {adaptation_task}, fraction:{dataset_percentage}%.csv"),
                mode='a', index_label='Epoch')


if __name__ == '__main__':
    main()
