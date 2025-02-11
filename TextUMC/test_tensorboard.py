"""Test script to verify TensorBoard integration"""

from model import TextUMC, Evidence, umc_train
import os
from datetime import datetime
import torch


def test_tensorboard():
    # Create test data
    test_evidences = [
        Evidence(evidence_id=str(i), content=f"Test evidence {i}") for i in range(10)
    ]

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextUMC().to(device)

    # Create log directory
    run_dir = os.path.join(
        "outputs", "tensorboard_test", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training model with TensorBoard logging to: {log_dir}")

    # Train model with TensorBoard logging
    _, metrics = umc_train(
        model=model,
        evidences=test_evidences,
        num_clusters=3,
        batch_size=2,
        num_epochs=5,
        learning_rate=0.001,
        log_dir=log_dir,
    )

    print("\nTraining complete. To view the training progress:")
    print(f"1. Logs saved to: {log_dir}")
    print("2. Run: tensorboard --logdir TextUMC/outputs/tensorboard_test")
    print("3. Open the provided URL (usually http://localhost:6006) in your browser")
    print("\nYou should see three metrics plotted over time:")
    print("- Loss/Total: Overall training loss")
    print("- Loss/Unsupervised: Contrastive loss component")
    print("- Loss/Supervised: Supervised loss component")


if __name__ == "__main__":
    test_tensorboard()
