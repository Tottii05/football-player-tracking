import time
import numpy as np
import torch
from rfdetr import RFDETRNano

def measure_inference(model, image, repetitions=20):
    """Mide el tiempo medio de inferencia y el pico de memoria GPU."""

    # Calentamiento: evita medir la inicialización inicial.
    for _ in range(3):
        model.predict(image, threshold=0.3)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for _ in range(repetitions):
        model.predict(image, threshold=0.3)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    average_ms = (elapsed / repetitions) * 1000
    fps = repetitions / elapsed
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    return average_ms, fps, peak_memory

def main():
    print("=" * 55)
    print("BENCHMARK RF-DETR NANO")
    print("=" * 55)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    image = np.zeros((640, 640, 3), dtype=np.uint8)

    print("\nInicializando RF-DETR Nano...")
    model = RFDETRNano()

    print("\n--- CONFIGURACIÓN FP32 ---")

    fp32_ms, fp32_fps, fp32_memory = measure_inference(model, image)

    print(f"Tiempo medio: {fp32_ms:.2f} ms")
    print(f"FPS: {fp32_fps:.2f}")
    print(f"Pico de memoria: {fp32_memory:.2f} GB")

    print("\n--- CONFIGURACIÓN FP16 ---")

    model.inference(dtype=torch.float16)
    fp16_ms, fp16_fps, fp16_memory = measure_inference(model, image)

    print(f"Tiempo medio: {fp16_ms:.2f} ms")
    print(f"FPS: {fp16_fps:.2f}")
    print(f"Pico de memoria: {fp16_memory:.2f} GB")

    print("\n--- COMPARACIÓN ---")

    print(f"FP32: {fp32_fps:.2f} FPS")
    print(f"FP16: {fp16_fps:.2f} FPS")

    if fp16_fps > fp32_fps:
        improvement = ((fp16_fps / fp32_fps) - 1) * 100
        print(f"Mejora de velocidad: {improvement:.2f}%")
    else:
        print("FP16 no ha mejorado la velocidad en esta prueba.")

    print("\nBENCHMARK COMPLETADO")

if __name__ == "__main__":
    main()