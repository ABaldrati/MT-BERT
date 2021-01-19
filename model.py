from collections import defaultdict
from typing import List, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss
from transformers import BertTokenizer, BertModel

from task import Task


class SSCModule(nn.Module):  # Single sentence classification
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return torch.nn.Softmax(self.output_layer(x))


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
            nn.Linear(4 * hidden_size, output_classes),
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


class MT_BERT(nn.Module):
    def __init__(self, bert_pretrained_model="bert-base-uncased"):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        self.bert = BertModel.from_pretrained(bert_pretrained_model)
        self.hidden_size = self.bert.config.hidden_size
        k_steps = self.bert.config.num_hidden_layers

        # Single-Sentence Classification modules
        self.CoLa = SSCModule(self.hidden_size, dropout_prob=0.05)
        self.SST_2 = SSCModule(self.hidden_size)

        # Pairwise Text Similarity module
        self.STS_B = PTSModule(self.hidden_size)

        # Pairwise Text Classification
        self.MNLI = PTCModule(self.hidden_size, k_steps, output_classes=Task.MNLI.num_classes(), dropout_prob=0.3,
                              stochastic_prediction_dropout_prob=0.3)
        self.RTE = PTCModule(self.hidden_size, k_steps, output_classes=Task.RTE.num_classes())
        self.WNLI = PTCModule(self.hidden_size, k_steps, output_classes=Task.WNLI.num_classes())
        self.QQP = PTCModule(self.hidden_size, k_steps, output_classes=Task.QQP.num_classes())
        self.MRPC = PTCModule(self.hidden_size, k_steps, output_classes=Task.MRPC.num_classes())
        self.SNLI = PTCModule(self.hidden_size, k_steps, output_classes=Task.SNLI.num_classes())
        self.SciTail = PTCModule(self.hidden_size, k_steps, output_classes=Task.SciTail.num_classes())

        # Pairwise Ranking
        self.QNLI = PRModule(self.hidden_size)

    def forward(self, x, task: Task):
        tokenized_input = self.tokenizer(x, padding=True, return_tensors='pt')
        bert_output = self.bert(**tokenized_input).last_hidden_state
        cls_embedding = bert_output[:, 0, :]
        if task == Task.CoLA:
            return self.CoLa(cls_embedding)
        elif task == Task.SST_2:
            return self.SST_2(cls_embedding)
        elif task == Task.STS_B:
            return self.STS_B(cls_embedding)
        elif task == Task.MNLI:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.MNLI(premises, hypotheses)
        elif task == Task.RTE:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.RTE(premises, hypotheses)
        elif task == Task.WNLI:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.WNLI(premises, hypotheses)
        elif task == Task.QQP:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.QQP(premises, hypotheses)
        elif task == Task.MRPC:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.MRPC(premises, hypotheses)
        elif task == Task.SNLI:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.SNLI(hypotheses, premises)
        elif task == Task.SciTail:
            hypotheses, premises = self.preprocess_PTC_input(bert_output, tokenized_input)
            return self.SciTail(hypotheses, premises)
        elif task == Task.QNLI:
            return self.QNLI(cls_embedding)

    @staticmethod
    def loss_for_task(t: Task):
        losses = {
            Task.CoLA: CrossEntropyLoss(),
            Task.SST_2: CrossEntropyLoss(),
            Task.STS_B: MSELoss(),
            Task.MNLI: CrossEntropyLoss(),
            Task.WNLI: CrossEntropyLoss(),
            Task.QQP: CrossEntropyLoss(),
            Task.MRPC: CrossEntropyLoss(),
            Task.QNLI: BCELoss(),
            Task.SNLI: CrossEntropyLoss(),
            Task.SciTail: CrossEntropyLoss()
        }

        return losses[t]

    def preprocess_PTC_input(self, bert_output, tokenized_input):
        # TODO: Remove trailing and heading zeros after masking
        mask_premises = tokenized_input.attention_mask * torch.logical_not(tokenized_input.token_type_ids)
        premises = mask_premises.unsqueeze(2).repeat(1, 1, self.hidden_size) * bert_output
        mask_hypotheses = tokenized_input.attention_mask * tokenized_input.token_type_ids
        hypotheses = mask_hypotheses.unsqueeze(2).repeat(1, 1, self.hidden_size) * bert_output
        return hypotheses, premises


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
