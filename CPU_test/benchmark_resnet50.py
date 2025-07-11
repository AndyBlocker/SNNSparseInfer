#!/usr/bin/env python3
import torch
import torchvision.models as models
import time
import numpy as np
import argparse
import sys

def warm_up(model, input_tensor, device, num_warmup=10):
    """预热模型"""
    print(f"Warming up on {device}...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark_fps(model, input_tensor, device, num_iterations=100):
    """测试FPS"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
    
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    return fps, avg_time, std_time

def test_gpu_backend():
    """测试GPU backend"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU test.")
        return None
    
    print("\n" + "="*50)
    print("Testing GPU Backend (CUDA)")
    print("="*50)
    
    device = torch.device('cuda')
    model = models.resnet50(pretrained=True).to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # 预热
    warm_up(model, input_tensor, device)
    
    # 测试FPS
    fps, avg_time, std_time = benchmark_fps(model, input_tensor, device)
    
    print(f"GPU Results:")
    print(f"  Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    
    return {'backend': 'GPU', 'fps': fps, 'avg_time': avg_time, 'std_time': std_time}

def test_mkl_backend():
    """测试MKL backend (CPU)"""
    print("\n" + "="*50)
    print("Testing MKL Backend (CPU)")
    print("="*50)
    
    # 设置MKL线程数
    torch.set_num_threads(torch.get_num_threads())
    print(f"Using {torch.get_num_threads()} CPU threads")
    
    device = torch.device('cpu')
    model = models.resnet50(pretrained=True).to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # 预热
    warm_up(model, input_tensor, device)
    
    # 测试FPS
    fps, avg_time, std_time = benchmark_fps(model, input_tensor, device)
    
    print(f"MKL Results:")
    print(f"  Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    
    return {'backend': 'MKL', 'fps': fps, 'avg_time': avg_time, 'std_time': std_time}

def main():
    parser = argparse.ArgumentParser(description='Benchmark ResNet50 inference FPS')
    parser.add_argument('--iterations', type=int, default=100, 
                       help='Number of inference iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Test GPU backend only')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Test CPU/MKL backend only')
    
    args = parser.parse_args()
    
    print("ResNet50 Inference Benchmark")
    print(f"Batch size: 1")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    
    # 检查PyTorch版本和MKL支持
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MKL available: {torch.backends.mkl.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    results = []
    
    # 测试GPU backend
    if not args.cpu_only:
        gpu_result = test_gpu_backend()
        if gpu_result:
            results.append(gpu_result)
    
    # 测试MKL backend  
    if not args.gpu_only:
        mkl_result = test_mkl_backend()
        if mkl_result:
            results.append(mkl_result)
    
    # 打印对比结果
    if len(results) > 1:
        print("\n" + "="*50)
        print("Comparison Results")
        print("="*50)
        
        gpu_fps = next((r['fps'] for r in results if r['backend'] == 'GPU'), None)
        mkl_fps = next((r['fps'] for r in results if r['backend'] == 'MKL'), None)
        
        if gpu_fps and mkl_fps:
            speedup = gpu_fps / mkl_fps
            print(f"GPU FPS: {gpu_fps:.2f}")
            print(f"MKL FPS: {mkl_fps:.2f}")
            print(f"GPU Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()