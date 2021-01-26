import hashlib
import operator
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import scipy
import torch
from tqdm import tqdm

from model import MT_BERT
from task import define_dataset_config, Task, define_tasks_config

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
    args = parser.parse_args()

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)

    model = MT_BERT()
    model.to(device)

    saved_model = torch.load(args.model, map_location=device)
    model.load_state_dict(saved_model['model_state_dict'])
    training_start = saved_model["training_start"]

    results_folder = Path(f"results_{training_start}")
    results_folder.mkdir(exist_ok=True)

    model.eval()
    test_results = {}
    with torch.no_grad():
        task_bar = tqdm(Task)
        for task in task_bar:
            task_bar.set_description(task.name)
            test_loader = tasks_config[task]["test_loader"]

            task_predicted_labels = torch.empty(0, device=device)
            task_labels = torch.empty(0, device=device)
            for test_data in test_loader:
                data_columns = [col for col in tasks_config[task]["columns"] if col != "label"]
                input_data = list(zip(*(test_data[col] for col in data_columns)))
                label = test_data["label"].to(device)

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
                test_results[task.name, metric.__name__] = metric_result
                print(f"test_results[{task.name}, {metric.__name__}] = {test_results[task.name, metric.__name__]}")
    data_frame = pd.DataFrame(
        data=test_results)
    data_frame.to_csv(str(results_folder / f"test_results.csv"), mode='a', index_label='Epoch')


if __name__ == '__main__':
    main()
