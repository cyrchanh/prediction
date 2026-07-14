import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from iewc.importance import ImportanceEstimator


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_task_data(digit_pair, n_samples=500):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mask = (mnist.targets == digit_pair[0]) | (mnist.targets == digit_pair[1])
    x = mnist.data[mask].float() / 255.0
    y = (mnist.targets[mask] == digit_pair[1]).long()
    return x[:n_samples], y[:n_samples]


def train(model, x, y, epochs=5, lr=1e-3):
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def accuracy(model, x, y):
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        return (preds == y).float().mean().item()


criterion = nn.CrossEntropyLoss()


def get_importance(model, dataset):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = ImportanceEstimator(kind="ief_diag", tau=0.25).compute(
        model=model, criterion=criterion, optimizer=optimizer,
        dataset=dataset, device=torch.device("cpu"), batch_size=32,
    )
    return {k: v.data.clone() for k, v in result.importances.items()}


def get_gradient_at_point(model, x, y):
    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def curvature_conflict(fisher_first, grad_second):
    total = 0.0
    for name in fisher_first:
        total += (fisher_first[name] * grad_second[name].pow(2)).sum().item()
    return total


# ---- Data ----
x_a, y_a = load_task_data((0, 1))
x_b, y_b = load_task_data((2, 3))

# ---- Direction 1: train on A, evaluate B's disruption to A ----
torch.manual_seed(0)
probe_a = train(MLP(), x_a, y_a)
task_ids_a = torch.zeros(len(y_a), dtype=torch.long)
dataset_a = TensorDataset(x_a, y_a, task_ids_a)
importance_a = get_importance(probe_a, dataset_a)
grad_b_at_a = get_gradient_at_point(probe_a, x_b, y_b)
conflict_b_disrupts_a = curvature_conflict(importance_a, grad_b_at_a)

# ---- Direction 2: train on B, evaluate A's disruption to B ----
torch.manual_seed(0)
probe_b = train(MLP(), x_b, y_b)
task_ids_b = torch.zeros(len(y_b), dtype=torch.long)
dataset_b = TensorDataset(x_b, y_b, task_ids_b)
importance_b = get_importance(probe_b, dataset_b)
grad_a_at_b = get_gradient_at_point(probe_b, x_a, y_a)
conflict_a_disrupts_b = curvature_conflict(importance_b, grad_a_at_b)

print(f"\nPredicted disruption to A if B trained next: {conflict_b_disrupts_a:.6f}")
print(f"Predicted disruption to B if A trained next: {conflict_a_disrupts_b:.6f}")
predicted_safer = "A->B" if conflict_b_disrupts_a < conflict_a_disrupts_b else "B->A"
print(f"Prediction: {predicted_safer} ordering is SAFER (less forgetting expected)")


# ---- Actually run both orderings for real ----
def run_sequence(first_data, second_data, label):
    model = MLP()
    x1, y1 = first_data
    x2, y2 = second_data
    train(model, x1, y1)
    acc_before = accuracy(model, x1, y1)
    train(model, x2, y2)
    acc_after = accuracy(model, x1, y1)
    forgetting = acc_before - acc_after
    print(f"{label}: before={acc_before:.3f}, after={acc_after:.3f}, forgetting={forgetting:.3f}")
    return forgetting

print("\n=== Actual sequential training results ===")
forgetting_ab = run_sequence((x_a, y_a), (x_b, y_b), "A then B (forgetting on A)")
forgetting_ba = run_sequence((x_b, y_b), (x_a, y_a), "B then A (forgetting on B)")

actual_safer = "A->B" if forgetting_ab < forgetting_ba else "B->A"
print(f"\nPredicted safer ordering: {predicted_safer}")
print(f"Actual safer ordering:    {actual_safer}")
print(f"Match: {predicted_safer == actual_safer}")
