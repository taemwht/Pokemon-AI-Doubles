"""
Train a PyTorch MLP to predict win probability (Target) from embed_battle vectors.
"""
import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "processed_training_data.csv"
MODEL_PATH = "vgc_value_model.pth"
SCALER_PATH = "vgc_value_scaler.pkl"
TRAIN_SIZE = 0.8
RANDOM_STATE = 42
EPOCHS = 50


class ValueMLP(nn.Module):
    """MLP: input -> 128 -> 64 -> 1 (Sigmoid)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    if "Target" not in df.columns:
        raise ValueError("CSV must have a 'Target' column")
    X = df.drop(columns=["Target"]).values.astype("float32")
    y = df["Target"].values.astype("float32").reshape(-1, 1)

    print("Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueMLP(n_features).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t.squeeze(-1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss = {loss.item():.4f}")

    print("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        proba = model(X_test_t)
        pred = (proba >= 0.5).float()
        correct = (pred == y_test_t.squeeze(-1)).float().sum().item()
    accuracy = correct / len(y_test)
    print(f"Test accuracy: {accuracy:.4f} ({correct:.0f}/{len(y_test)})")

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model weights to {MODEL_PATH} and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
