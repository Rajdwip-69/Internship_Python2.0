import torch
import torch.nn as nn
import torch.optim as optim

# Sample dataset (very small)
text = "i love python i love coding python is fun coding is fun"

# Create vocabulary
words = text.split()
vocab = list(set(words))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Prepare training data
pairs = []
for i in range(len(words) - 1):
    pairs.append((words[i], words[i+1]))

# Model
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 10)
        self.linear = nn.Linear(10, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

model = SimpleLLM(len(vocab))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    total_loss = 0
    for w1, w2 in pairs:
        input_ix = torch.tensor([word_to_ix[w1]])
        target_ix = torch.tensor([word_to_ix[w2]])

        optimizer.zero_grad()
        output = model(input_ix)
        loss = loss_function(output, target_ix)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

# Test
def predict(word):
    with torch.no_grad():
        input_ix = torch.tensor([word_to_ix[word]])
        output = model(input_ix)
        _, predicted_ix = torch.max(output, 1)
        return ix_to_word[predicted_ix.item()]

# Try it
print(predict("i"))
print(predict("python"))