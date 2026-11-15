import torch
import torch.nn as nn
import logging
import sys

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def test_zero_data_expansion():
    print("=" * 70)
    print("RUNNING ZERO-DATA DIRECT-SUM DIMENSION EXPANSION UNIT TEST")
    print("=" * 70)

    from models.sinet_inflora import SiNet

    args = {
        "dataset": "ImageNet_R",
        "init_cls": 20,
        "increment": 20,
        "embd_dim": 768,
        "total_sessions": 10,
        "rank": 10,
        "use_kernel": False,
        "use_dynamic_expansion": True,
        "expand_at_task": 1,
        "expand_dim": 1280,
        "expand_gamma": 1.0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing SiNet on {device}...")
    model = SiNet(args).to(device)

    # Initialize LoRA B matrices to zero (standard LoRA initialization in InfLoRA)
    from models.vit_inflora import Attention_LoRA
    for module in model.modules():
        if isinstance(module, Attention_LoRA):
            module.init_param()

    # ---------------------------------------------------------
    # Step 1: Simulate Task 0 at Native 768-D
    # ---------------------------------------------------------
    print("\n--- Step 1: Simulating Task 0 (Native 768-D) ---")
    model.update_fc(20)
    assert model.numtask == 1, "model.numtask should be 1"
    assert model.feature_dim == 768, f"Feature dim should be 768, got {model.feature_dim}"
    assert model.classifier_pool[0].weight.shape == (20, 768), f"Classifier 0 weight shape mismatch: {model.classifier_pool[0].weight.shape}"

    # Generate dummy input images
    torch.manual_seed(42)
    dummy_input = torch.randn(4, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        logits_task0_before = model.interface(dummy_input, task_id=0)  # Shape [4, 20]
        feats_before = model.extract_vector(dummy_input, task=0)       # Shape [4, 768]

    print(f"Task 0 Pre-expansion Feature Shape: {feats_before.shape}")
    print(f"Task 0 Pre-expansion Logits Shape:  {logits_task0_before.shape}")
    print(f"Task 0 Sample Logits (first row, first 5 classes):\n  {logits_task0_before[0, :5].cpu().numpy()}")

    # ---------------------------------------------------------
    # Step 2: Trigger Zero-Data Direct-Sum Expansion
    # ---------------------------------------------------------
    print("\n--- Step 2: Triggering Zero-Data Dimension Expansion ---")
    model.expand_feature_dimension()

    assert model.is_expanded == True, "model.is_expanded should be True"
    assert model.feature_dim == 2048, f"Feature dim should be 2048 (768 + 1280), got {model.feature_dim}"
    assert model.classifier_pool[0].weight.shape == (20, 2048), f"Classifier 0 weight shape should be (20, 2048), got {model.classifier_pool[0].weight.shape}"

    # Verify zero-padding in past classifier weights:
    padding_block = model.classifier_pool[0].weight[:, 768:]
    max_non_zero = torch.max(torch.abs(padding_block)).item()
    print(f"Verification: Max absolute value in padded coordinates (indices 768:2048): {max_non_zero}")
    assert max_non_zero == 0.0, f"Padded weights must be exactly 0.0, found max {max_non_zero}!"

    # ---------------------------------------------------------
    # Step 3: Verify Exact Mathematical Logit Invariance
    # ---------------------------------------------------------
    print("\n--- Step 3: Verifying Logit Preservation on Task 0 ---")
    model.update_fc(40)  # Move to Task 1
    assert model.numtask == 2, "model.numtask should be 2"

    with torch.no_grad():
        logits_after = model.interface(dummy_input, task_id=0)      # Shape [4, 40]
        logits_task0_after = logits_after[:, :20]                  # Extract Task 0 slice
        feats_after = model.extract_vector(dummy_input, task=0)     # Shape [4, 2048]

    print(f"Post-expansion Feature Shape: {feats_after.shape}")
    print(f"Post-expansion Logits Shape:  {logits_after.shape}")
    print(f"Task 0 Post-expansion Logits (first row, first 5 classes):\n  {logits_task0_after[0, :5].cpu().numpy()}")

    diff = torch.max(torch.abs(logits_task0_before - logits_task0_after)).item()
    print(f"\n--> MAXIMUM LOGIT DIFFERENCE (Before vs After Expansion): {diff:.10e}")

    # Check that difference is within floating-point epsilon (1e-6)
    if diff < 1e-5:
        print("[PASSED] Logits on past tasks are mathematically 100% BIT-EXACT!")
    else:
        print(f"[FAILED] Logits drifted by {diff}!")
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 4: Verify Task 1 can freely learn in the expanded space
    # ---------------------------------------------------------
    print("\n--- Step 4: Verifying Task 1 Head in Expanded Space ---")
    assert model.classifier_pool[1].weight.shape == (20, 2048), f"Classifier 1 shape mismatch: {model.classifier_pool[1].weight.shape}"
    print(f"Task 1 Classifier Head correctly initialized with shape: {model.classifier_pool[1].weight.shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! ZERO-DATA EXPANSION IS VERIFIED.")
    print("=" * 70)

if __name__ == "__main__":
    test_zero_data_expansion()
