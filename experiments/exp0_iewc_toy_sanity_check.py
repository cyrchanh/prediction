import torch
from torch import nn
from torch.utils.data import TensorDataset
from iewc.importance import ImportanceEstimator

x = torch.tensor([[2.0, 0.0], [1.5, 0.5], [-2.0, 0.0], [-1.5, -0.5]])
y = torch.tensor([0, 0, 1, 1])
task = torch.zeros(len(y), dtype=torch.long)
dataset = TensorDataset(x, y, task)

model = nn.Linear(2, 2, bias=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for kind in ["ef", "ewc_dr", "ief_diag"]:
    result = ImportanceEstimator(kind=kind, tau=0.25).compute(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=2,
    )
    total_importance = sum(v.data.sum().item() for v in result.importances.values())
    print(f"{kind:10s} -> total importance = {total_importance:.4f}")
