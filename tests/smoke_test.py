import torch
import numpy as np
from src.features.extraction import extract_1d_features, extract_2d_features, SR
from src.models.registry import get_model

def test_feature_extraction():
    print("Testing Feature Extraction...")
    # 10 second pure noise
    wav = np.random.uniform(-1, 1, SR * 10).astype(np.float32)
    
    # 1D
    feat1d = extract_1d_features(wav, augment=False)
    print(f"1D Feature shape: {feat1d.shape} (Expected: (7, 432))")
    assert feat1d.shape == (7, 432)
    
    # 2D
    feat2d = extract_2d_features(wav)
    print(f"2D Feature shape: {feat2d.shape} (Expected: (1, 84, 432))")
    assert feat2d.shape == (1, 84, 432)
    print("Feature Extraction: PASS")

def test_models():
    print("\nTesting Models...")
    device = "cpu"
    
    # cnn_1d
    model = get_model("cnn_1d").to(device)
    x = torch.randn(2, 7, 432).to(device)
    p, s, d = model(x)
    print(f"cnn_1d output shapes: {p.shape}, {s.shape}, {d.shape}")
    assert p.shape == (2, 3) and s.shape == (2,) and d.shape == (2,)
    
    # cnn_2d
    model = get_model("cnn_2d").to(device)
    x = torch.randn(2, 1, 84, 432).to(device)
    p, s, d = model(x)
    print(f"cnn_2d output shapes: {p.shape}, {s.shape}, {d.shape}")
    assert p.shape == (2, 3) and s.shape == (2,) and d.shape == (2,)
    
    # attn_1d
    model = get_model("attn_1d").to(device)
    x = torch.randn(2, 7, 432).to(device)
    p, s, d = model(x)
    print(f"attn_1d output shapes: {p.shape}, {s.shape}, {d.shape}")
    assert p.shape == (2, 3) and s.shape == (2,) and d.shape == (2,)
    
    # attn_2d
    model = get_model("attn_2d").to(device)
    x = torch.randn(2, 1, 84, 432).to(device)
    p, s, d = model(x)
    print(f"attn_2d output shapes: {p.shape}, {s.shape}, {d.shape}")
    assert p.shape == (2, 3) and s.shape == (2,) and d.shape == (2,)
    print("Models: PASS")

if __name__ == "__main__":
    try:
        test_feature_extraction()
        test_models()
        print("\nALL SMOKE TESTS PASSED!")
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
