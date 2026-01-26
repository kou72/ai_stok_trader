"""
GPU利用可能性の確認
"""
import torch
import sys

print("=" * 80)
print("GPU確認スクリプト")
print("=" * 80)

# 1. PyTorchのバージョン
print(f"\n【PyTorchバージョン】")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA利用可能: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n[ERROR] CUDAが利用できません")
    print("   原因:")
    print("   - CPU版PyTorchがインストールされている")
    print("   - CUDAドライバーがインストールされていない")
    print("   - GPUがない")
    sys.exit(1)

# 2. CUDA情報
print(f"\n【CUDA情報】")
print(f"CUDAバージョン: {torch.version.cuda}")
print(f"cuDNNバージョン: {torch.backends.cudnn.version()}")
print(f"利用可能なGPU数: {torch.cuda.device_count()}")

# 3. GPU情報
print(f"\n【GPU情報】")
for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}:")
    print(f"  名前: {torch.cuda.get_device_name(i)}")
    print(f"  メモリ容量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"  計算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

# 4. 現在のデバイス
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n【使用デバイス】")
print(f"デバイス: {device}")

# 5. 簡単な演算テスト
print(f"\n【GPU演算テスト】")
try:
    # CPUでテスト
    x_cpu = torch.randn(1000, 1000)
    y_cpu = torch.randn(1000, 1000)
    
    import time
    start = time.time()
    z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU演算時間: {cpu_time:.4f}秒")
    
    # GPUでテスト
    x_gpu = torch.randn(1000, 1000).to(device)
    y_gpu = torch.randn(1000, 1000).to(device)
    
    # ウォームアップ
    _ = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    z_gpu = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU演算時間: {gpu_time:.4f}秒")
    print(f"高速化率: {cpu_time/gpu_time:.2f}x")
    
    print("\n[OK] GPUは正常に動作しています！")
    
except Exception as e:
    print(f"\n[ERROR] GPU演算エラー: {e}")

# 6. メモリ情報
print(f"\n【GPUメモリ情報】")
print(f"割り当て済み: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"キャッシュ済み: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n" + "=" * 80)