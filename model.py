from collections import defaultdict
from typing import List, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss

from transformers import BertTokenizer, BertModel


class SSCModule(nn.Module):  # Single sentence classification
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return F.softmax(self.output_layer(x))


class PTSModule(nn.Module):  # Pairwise text similarity
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.output_layer(x)


class PTCModule(nn.Module):  # Pariwise text classification
    def __init__(self, hidden_size, k_steps, output_classes, dropout_prob=0.1, stochastic_prediction_dropout_prob=0.1):
        super().__init__()
        self.stochastic_prediction_dropout = stochastic_prediction_dropout_prob
        self.k_steps = k_steps
        self.hidden_size = hidden_size
        self.output_classes = output_classes

        self.GRU = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=dropout_prob)

        self.W1 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout_prob))

        self.W2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_prob))

        self.W3 = nn.Sequential(
            nn.Linear(hidden_size, output_classes),
            nn.Dropout(dropout_prob))

    def forward(self, premises: torch.Tensor, hypotheses: torch.Tensor):
        batch_size = premises.size(0)
        output_probabilities = torch.zeros(batch_size, self.output_classes)

        flatten_hypotheses = hypotheses.view(-1, self.hidden_size)
        flatten_premise = premises.view(-1, self.hidden_size)

        alfas = F.softmax(self.W1(flatten_hypotheses).view(batch_size, - 1), -1)
        s_state = (alfas.unsqueeze(1) @ hypotheses)  # (Bs,1,hidden)

        layer_output = self.W2(flatten_premise).view(batch_size, -1, self.hidden_size)
        layer_output_transpose = torch.transpose(layer_output, 1, 2)

        actual_k = 0
        for k in range(self.k_steps):
            betas = F.softmax(s_state @ layer_output_transpose, -1)  # TODO check correctness
            x_input = betas @ premises
            _, s_state = self.GRU(x_input, s_state.transpose(0, 1))
            s_state = s_state.transpose(0, 1)
            concatenated_features = torch.cat([s_state, x_input, (s_state - x_input).abs(), x_input * s_state], -1)
            if torch.rand(()) > self.stochastic_prediction_dropout:
                output_probabilities += F.softmax(self.W3(concatenated_features), -1).squeeze()
                actual_k += 1

        return output_probabilities / actual_k


class PRModule(nn.Module):  # Pairwise ranking module
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return torch.sigmoid(self.output_layer(x))


def compute_qnli_batch_output(batch, class_label, model):
    questions = batch["question"]
    answers = batch["answer"]
    labels = batch["label"]

    relevant_answers = defaultdict(list)
    for question, answer, label in zip(questions, answers, labels):
        if class_label.int2str(label) == "entailment":
            relevant_answers[question].append(answer)

    relevance_scores = torch.empty(0)
    for question, answer, label in zip(questions, answers, labels):
        softmax_answers: List[Any] = answers
        for _ in range(len(relevant_answers[question])):
            softmax_answers.remove(answer)

        softmax_answers.append(answer)
        model_input = []
        for a in softmax_answers:
            model_input.append([question, a])

        model_output = model(model_input)
        relevance_scores = torch.stack((relevance_scores, torch.softmax(model_output, -1)[-1]))

    return relevance_scores
