import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import tqdm

# Load data
hiddens = torch.load('/scr/jphilipp/tstar/data/hiddens_train.pt')
n = int(0.9 * hiddens.shape[0])
final_hidden = hiddens[:n]
eval_hidden = hiddens[n:]

data = json.load(open('gsm_results_llama_0_shot_train.json'))
labels = [label['label'] for label in data[:n]]
eval_labels = [label['label'] for label in data[n:]]

breakpoint()
# Prepare training data with balanced classes
x, y = [], []
count_false, count_true = 0, 0
for i, label in enumerate(labels):
    if label and count_true < 1000:
        count_true += 1
        x.append(final_hidden[i])
        y.append(int(label))
    elif not label and count_false < 1000:
        count_false += 1
        x.append(final_hidden[i])
        y.append(int(label))

breakpoint()

x = torch.stack(x).float()
y = torch.tensor(y).float()


x_eval = torch.tensor(eval_hidden).float() 
y_eval = torch.tensor(eval_labels).float()

breakpoint()
torch.manual_seed(1337)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        return x

breakpoint()
input_dim = 1
model = Classifier(input_dim).to('cuda')
criterion = nn.BCELoss()  
optimizer = optim.AdamW(model.parameters(), lr=.1)  # Reduced learning rate
breakpoint()
def get_batch(batch_size=5):
    rand_idx = torch.randint(len(x), (batch_size,))
    xb = x[rand_idx]
    yb = y[rand_idx]
    return xb, yb

def evaluate(model, x_eval, y_eval):
    model.eval()
    with torch.no_grad():
        outputs = model(x_eval.to('cuda')).squeeze()
        predictions = torch.round(outputs)
        accuracy = (predictions == y_eval.to('cuda')).float().mean().item()
    return accuracy

# Training loop
num_epochs = 100
batch_size = 200
losses = []
accuracy = evaluate(model, x_eval, y_eval)
print(f'Evaluation Accuracy: {accuracy:.4f}')
breakpoint()
for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    x_train, y_train = get_batch(batch_size)
    outputs = model(x_train.to('cuda')).squeeze()
    breakpoint()
    loss = criterion(outputs, y_train.to('cuda'))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        

breakpoint()
accuracy = evaluate(model, x_eval, y_eval)
print(f'Final Evaluation Accuracy: {accuracy:.4f}')