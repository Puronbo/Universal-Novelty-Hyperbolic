"""
Direct tensor training of the hyperbolic manifold, bypassing the browser
entirely. Instead of storing node positions in JSON/localStorage, this
learns them as model weights and checkpoints them to a .pth file.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class HyperbolicMapper(nn.Module):
    def __init__(self, num_nodes: int = 100):
        super().__init__()
        # Raw weights representing manifold node positions
        self.weights = nn.Parameter(torch.randn(num_nodes, 2) * 0.01)

    def forward(self):
        # tanh keeps every node strictly inside the unit disk (radius < 1)
        return torch.tanh(self.weights)


def train(num_epochs: int = 1000, lr: float = 0.01, log_every: int = 100):
    model = HyperbolicMapper()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Same topic anchors as the Python engine and the web dashboard
    anchors = torch.tensor([
        [0.50, 0.50],    # physics_core
        [-0.60, 0.40],   # tech_infra
        [0.10, -0.65],   # human_culture
    ])

    print("Training AI directly on tensor flow...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        coords = model()

        # Pull every node toward its nearest anchor
        dists = torch.cdist(coords, anchors)
        nearest_dist, _ = dists.min(dim=1)
        loss = nearest_dist.pow(2).mean()

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "manifold_model.pth")
    print("Saved trained manifold to manifold_model.pth")
    return model


if __name__ == "__main__":
    train()
