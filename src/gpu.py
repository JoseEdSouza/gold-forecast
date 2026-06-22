"""Utilitários de configuração de GPU/CUDA para notebooks e scripts."""

import os
import sys


def setup_gpu() -> None:
    """Configura cuDNN e XLA antes de importar TensorFlow, e habilita memory growth na GPU.

    Deve ser chamada na primeira célula do notebook, antes de qualquer import do TensorFlow.
    Se os caminhos NVIDIA não forem encontrados no venv atual, imprime aviso mas não falha.
    """
    _py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _site = os.path.join(sys.prefix, "lib", _py, "site-packages", "nvidia")

    _cudnn_lib = os.path.join(_site, "cudnn", "lib")
    if os.path.isdir(_cudnn_lib):
        os.environ["LD_LIBRARY_PATH"] = _cudnn_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        print("cuDNN path:", _cudnn_lib)
    else:
        print("cuDNN não encontrado em", _cudnn_lib)

    _cuda_nvcc = os.path.join(_site, "cuda_nvcc")
    if os.path.isdir(_cuda_nvcc):
        xla_flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = xla_flags + f" --xla_gpu_cuda_data_dir={_cuda_nvcc}"
        print("cuda_nvcc path:", _cuda_nvcc)
    else:
        print("cuda_nvcc não encontrado em", _cuda_nvcc, "— instale nvidia-cuda-nvcc-cu12")


def configure_gpu_memory_growth() -> None:
    """Habilita memory growth em todas as GPUs disponíveis e imprime o resultado.

    Deve ser chamada após importar TensorFlow. Se não houver GPU, ou se a configuração
    falhar (driver incompatível, GPU ocupada, etc.), imprime aviso e segue na CPU.
    """
    import tensorflow as tf  # noqa: PLC0415

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Nenhuma GPU encontrada — treinando na CPU.")
        return

    enabled = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            enabled.append(gpu.name)
        except RuntimeError as e:
            # Ocorre se a GPU já foi inicializada antes desta chamada
            print(f"Aviso: não foi possível configurar memory growth em {gpu.name}: {e}")

    if enabled:
        print(f"GPU(s) configurada(s) com memory growth: {enabled}")
    else:
        print("GPUs encontradas mas nenhuma configurada — verifique os avisos acima.")
