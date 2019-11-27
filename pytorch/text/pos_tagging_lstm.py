import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

use_cuda = not False and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

lstm = nn.LSTM(3, 3)

inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]

# initiallize the hidden state

hidden = (autograd.Variable(torch.randn(1, 1, 3)),
    autograd.Variable(torch.randn((1, 1, 3))))

for i in inputs:
    # 하나씩 step
    # 각 step 후 hidden 은 hidden state 를 포함
    out, hidden = lstm(i.view(1, 1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1) #개별의 tensor 를 하나의 tensor
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))

out, hidden = lstm(inputs, hidden)

# 품사 태깅
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)

    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

word_to_idx = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

Embedding_dim = 6
hidden_dim = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        #initialize hidden state

        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim = 1)

        return tag_scores

model = LSTMTagger(Embedding_dim, hidden_dim, len(word_to_idx), len(tag_to_ix))
model = model.to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

inputs = prepare_sequence(training_data[0][0], word_to_idx)
inputs = inputs.cuda()

tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:

        model.zero_grad()

        model.hidden = model.init_hidden()

        sentence_in = prepare_sequence(sentence, word_to_idx)
        targets = prepare_sequence(tags, tag_to_ix)

        sentence_in = sentence_in.to(device)
        targets = targets.to(device)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_idx)
tag_scores = model(inputs.cuda())

print(tag_scores)
