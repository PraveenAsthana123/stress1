#!/usr/bin/env python3
"""
Test script using sample 100-row dataset.

This script validates the GenAI-RAG-EEG model using a small sample dataset
that is included in the repository for quick testing.

Usage:
    python test_sample_data.py

Output:
    - Model training on sample data
    - Classification metrics
    - Timestamps for all operations
"""

import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

def main():
    print("=" * 70)
    print("GenAI-RAG-EEG Sample Data Test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load sample data
    print("\n[1] Loading sample data...")
    X = np.load("data/sample_validation/X_sample_100.npy")
    y = np.load("data/sample_validation/y_sample_100.npy")

    with open("data/sample_validation/metadata_sample_100.json", 'r') as f:
        metadata = json.load(f)

    print(f"    Data shape: {X.shape}")
    print(f"    Labels: Baseline={np.sum(y==0)}, Stress={np.sum(y==1)}")
    print(f"    Source: {metadata.get('source', 'N/A')}")

    # Train-test split
    print("\n[2] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # Load model
    print("\n[3] Loading GenAI-RAG-EEG model...")
    from src.models.genai_rag_eeg import GenAIRAGEEG

    model = GenAIRAGEEG(n_channels=32, n_time_samples=512, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Device: {device}")

    # Training
    print("\n[4] Training model...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    best_acc = 0

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs['logits'], y_train_t)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            _, pred = torch.max(test_out['logits'], 1)
            acc = accuracy_score(y_test, pred.cpu().numpy())
            if acc > best_acc:
                best_acc = acc
        model.train()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:2d}/50 | Loss: {loss.item():.4f} | Test Acc: {acc*100:.1f}%")

    # Final evaluation
    print("\n[5] Final Evaluation...")
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        probs = test_out['probs'].cpu().numpy()
        _, pred = torch.max(test_out['logits'], 1)

    y_pred = pred.cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n    Results on Sample Data (n=100):")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {prec*100:.1f}%")
    print(f"    Recall:    {rec*100:.1f}%")
    print(f"    F1-Score:  {f1*100:.1f}%")
    print(f"    Best Acc:  {best_acc*100:.1f}%")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"               Baseline  Stress")
    print(f"    Actual Baseline  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"    Actual Stress    {cm[1,0]:4d}    {cm[1,1]:4d}")

    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "best_accuracy": best_acc
    }


if __name__ == "__main__":
    results = main()
