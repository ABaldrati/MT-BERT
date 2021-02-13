import hashlib
import operator
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from model import MT_BERT
from task import define_dataset_config, Task, define_tasks_config
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

    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=Task, nargs='+', default=list(Task))
    args = parser.parse_args()

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)

    model = MT_BERT()
    model.to(device)

    saved_model = torch.load(args.model, map_location=device)
    model.load_state_dict(saved_model['model_state_dict'])
    training_start = saved_model["training_start"]

    test_tasks = args.tasks

    results_folder = Path(f"results_{training_start}")
    results_folder.mkdir(exist_ok=True)
    if test_tasks == list(Task):
        glue_results_folder = Path(results_folder / f"glue_submission_GLUE_tasks")
    else:
        glue_results_folder = Path(results_folder / f"glue_submission_{test_tasks}")
    glue_results_folder.mkdir(exist_ok=True)

    model.eval()
    test_results = {}
    with torch.no_grad():
        with stream_redirect_tqdm() as orig_stdout:
            task_bar = tqdm(test_tasks, file=orig_stdout)
            for task in task_bar:
                task_bar.set_description(task.name)
                test_loader = tasks_config[task]["test_loader"]
                class_label = tasks_config[task]["test_dataset"].features['label']

                task_predicted_labels = torch.empty(0, device=device)
                task_labels = torch.empty(0, device=device)
                indexes = torch.empty(0, device=device)
                for test_data in test_loader:
                    data_columns = [col for col in tasks_config[task]["columns"] if col != "label"]
                    input_data = list(zip(*(test_data[col] for col in data_columns)))
                    label = test_data["label"].to(device)

                    if task != task.SciTail and task != task.SNLI:
                        indexes = torch.hstack((indexes, test_data['idx'].to(device)))

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
                if task == task.SciTail or task == task.SNLI:
                    for metric in metrics:
                        metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                        test_results[task.name, metric.__name__] = metric_result
                        print(
                            f"test_results[{task.name}, {metric.__name__}] = {test_results[task.name, metric.__name__]}")
                else:
                    if task in [Task.QNLI, Task.MNLIm, Task.MNLImm, Task.AX, Task.RTE]:
                        task_predicted_labels = class_label.int2str(task_predicted_labels)
                    elif task != Task.STS_B:
                        task_predicted_labels = task_predicted_labels.cpu().to(torch.int8)
                    else:
                        task_predicted_labels = task_predicted_labels.cpu()
                    data_frame = pd.DataFrame(data={'index': indexes.cpu().to(torch.int32),
                                                    'prediction': task_predicted_labels})

                    data_frame.to_csv(str(glue_results_folder / f"{task.value}.tsv"), sep='\t', index=False)
    if test_results:
        data_frame = pd.DataFrame(
            data=test_results, index=[0])
        data_frame.to_csv(str(results_folder / f"Scitail_snli_results.csv"), mode='a', index_label='Epoch')


if __name__ == '__main__':
    main()
