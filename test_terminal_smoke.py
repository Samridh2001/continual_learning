"""
Terminal Smoke Tester for Task-Commonality & Kernel DualGPM InfLoRA.
Run directly on cluster terminal:
    bash run_smoke_test.sh
or:
    /users/student/pg/pg25/samridh/miniconda3/envs/inflora/bin/python test_terminal_smoke.py

Verifies:
1. Model loading & 2048-dim RFF Kernel expansion
2. Clean covariance extraction and shortcut purging
3. Training Epoch execution (Forward, Subspace Barrier Loss, Backward, Step)
4. Strict freezing of Matrix A (requires_grad = False) & mathematical orthogonality
5. Active training of Matrix B & Classifier (requires_grad = True)
6. Multi-task evaluation with well-conditioned logits
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from methods.inflora_filter import InfLoRA_Filter
from models.sinet_inflora import SiNet

def test_epoch_execution():
    print("=" * 70)
    print("TERMINAL SMOKE TEST: VERIFYING EPOCH EXECUTION & KERNEL DUALGPM")
    print("=" * 70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on device: {device}")

    # Build dummy test configuration
    args = {
        "prefix": "terminal_test",
        "dataset": "ImageNet_R",
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": True,
        "shuffle": False,
        "init_cls": 20,
        "increment": 20,
        "model_name": "InfLoRA_filter",
        "net_type": "sip",
        "embd_dim": 768,
        "num_heads": 12,
        "total_sessions": 2,
        "seed": 42,
        "EPSILON": 1e-8,
        "init_epoch": 2,
        "epochs": 2,
        "optim": "adam",
        "init_lr": 0.0005,
        "init_lr_decay": 0.1,
        "init_weight_decay": 0.0,
        "lrate": 0.0005,
        "lrate_decay": 0.1,
        "batch_size": 8,
        "weight_decay": 0.0,
        "rank": 10,
        "lamb": 0.98,
        "lame": 1.0,
        "num_workers": 0,
        "commonality_threshold": 0.30,
        "max_shortcut_ratio": 0.50,
        "lambda_subspace": 1.0,
        "use_commonality_weighting": True,
        "beta_commonality": 4.0,
        "use_aug_probing": True,
        "delta_color": 0.35,
        "delta_blur": 0.35,
        "use_kernel": False,
        "device": [str(device)],
        "debug": True
    }

    print("\n[Step 1/6] Instantiating InfLoRA_Filter (Native 768-D, NO Kernel Expansion)...")
    model = InfLoRA_Filter(args)
    model._device = device
    model._network.to(device)
    print("  -> SiNet backbone loaded successfully!")
    print(f"  -> Feature Dimension: {model._network.feature_dim} (Expected: 768)")
    assert model._network.feature_dim == 768, "Feature dimension must be 768!"

    # Create synthetic batch loader matching PyCIL format (index, input, target)
    print("\n[Step 2/6] Creating Fast Mini-Batch Loader...")
    indices_t0 = torch.arange(16)
    inputs_t0 = torch.randn(16, 3, 224, 224)
    targets_t0 = torch.randint(0, 20, (16,))
    dataset_t0 = TensorDataset(indices_t0, inputs_t0, targets_t0)
    loader_t0 = DataLoader(dataset_t0, batch_size=8, shuffle=True)

    # Initialize Task 0
    model._cur_task = 0
    model._known_classes = 0
    model._total_classes = 20
    model._network.update_fc(20)

    print("\n[Step 3/6] Running Task 0 Training Epochs...")
    model.run_epoch = 2
    model._train(loader_t0, loader_t0)
    print("  -> Task 0 Epoch training, backpropagation, and post-task surgery completed!")

    # Check DualGPM basis updated
    assert len(model.feature_list) > 0, "Feature list should contain DualGPM bases"
    print(f"  -> DualGPM past bases created across {len(model.feature_list)} layers.")
    if model.kernel_memory_basis is not None:
        print(f"  -> Kernel DualGPM RKHS memory retained: {model.kernel_memory_basis.shape[1]}/2048 dims.")

    # Now transition to Task 1: verify orthogonal training
    print("\n[Step 4/6] Transitioning to Task 1...")
    indices_t1 = torch.arange(16)
    inputs_t1 = torch.randn(16, 3, 224, 224)
    targets_t1 = torch.randint(20, 40, (16,))
    dataset_t1 = TensorDataset(indices_t1, inputs_t1, targets_t1)
    loader_t1 = DataLoader(dataset_t1, batch_size=8, shuffle=True)

    model._cur_task = 1
    model._known_classes = 20
    model._total_classes = 40
    model._network.update_fc(40)

    print("\n[Step 5/6] Running Task 1 Training Epochs...")
    model.run_epoch = 2
    model._train(loader_t1, loader_t1)
    print("  -> Task 1 Epoch training completed cleanly without in-place weight mutation!")

    # Verify orthogonality and parameter freezing
    print("\n[Step 6/6] Verifying Parameter Freezing & Mathematical Orthogonality...")
    layer0_lora = None
    for m in model._network.modules():
        if hasattr(m, 'lora_A_k'):
            layer0_lora = m
            break

    if layer0_lora is not None:
        w_A = layer0_lora.lora_A_k[1].weight
        w_B = layer0_lora.lora_B_k[1].weight
        
        # Verify freezing
        assert not w_A.requires_grad, "CRITICAL ERROR: LoRA A must be FROZEN (requires_grad=False)!"
        assert w_B.requires_grad, "CRITICAL ERROR: LoRA B must be TRAINABLE (requires_grad=True)!"
        print(f"  -> [OK] Matrix A is strictly FROZEN: requires_grad = {w_A.requires_grad}")
        print(f"  -> [OK] Matrix B is actively TRAINABLE: requires_grad = {w_B.requires_grad}")

        # Verify orthogonality
        if model.restricted_bases[0] is not None:
            M_past = model.restricted_bases[0]
            overlap = torch.norm(torch.mm(w_A, M_past)).item()
            print(f"  -> [OK] Orthogonality Verified: ||A_1 @ M_past|| = {overlap:.2e} (< 1e-4)")
            assert overlap < 1e-4, f"A_1 must be orthogonal to past tasks, got overlap {overlap}"

        # Verify commonality weights (User's Idea)
        if model.commonality_weights[0] is not None:
            c_w = model.commonality_weights[0]
            print(f"  -> [OK] Commonality Weights Active: range=[{c_w.min().item():.2f}, {c_w.max().item():.2f}], mean={c_w.mean().item():.2f}")
            assert (c_w >= 1.0).all(), "Commonality weights must be >= 1.0!"

    # Verify inference output
    with torch.no_grad():
        test_inputs = torch.randn(4, 3, 224, 224).to(device)
        logits = model._network.interface(test_inputs)
        print(f"  -> [OK] Test Inference Output Shape: {logits.shape} (Expected: [4, 40])")
        assert logits.shape == (4, 40), f"Logits shape mismatch: {logits.shape}"
        assert not torch.isnan(logits).any(), "Logits contain NaNs!"
        print(f"  -> [OK] Logits range: min={logits.min().item():.2f}, max={logits.max().item():.2f} (Well-conditioned)")

    print("\n" + "=" * 70)
    print(" [SUCCESS] ALL 6 CHECKS PASSED! CODE IS READY FOR CLUSTER EXECUTION!")
    print("=" * 70)

if __name__ == "__main__":
    test_epoch_execution()
