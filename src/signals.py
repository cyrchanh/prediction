import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from iewc.importance import ImportanceEstimator

criterion = nn.CrossEntropyLoss()


def get_importance(model, x, y, kind="ief_diag", tau=0.25):
    task_ids = torch.zeros(len(y), dtype=torch.long)
    dataset = TensorDataset(x, y, task_ids)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # required arg, unused
    result = ImportanceEstimator(kind=kind, tau=tau).compute(
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
    """How much does grad_second land on directions fisher_first marks important?"""
    total = 0.0
    for name in fisher_first:
        total += (fisher_first[name] * grad_second[name].pow(2)).sum().item()
    return total


def predict_safer_ordering(model_a, x_a, y_a, x_b, y_b, model_b, kind="ief_diag"):
    """
    Both conflict scores are computed from a SHARED trajectory per direction
    (see Section 5.3 — this is the fix for the independent-initialisation pitfall).
    """
    importance_a = get_importance(model_a, x_a, y_a, kind=kind)
    grad_b_at_a = get_gradient_at_point(model_a, x_b, y_b)
    conflict_b_disrupts_a = curvature_conflict(importance_a, grad_b_at_a)

    importance_b = get_importance(model_b, x_b, y_b, kind=kind)
    grad_a_at_b = get_gradient_at_point(model_b, x_a, y_a)
    conflict_a_disrupts_b = curvature_conflict(importance_b, grad_a_at_b)

    predicted_safer = "B_then_A" if conflict_b_disrupts_a < conflict_a_disrupts_b else "A_then_B"
    return predicted_safer, conflict_b_disrupts_a, conflict_a_disrupts_b
