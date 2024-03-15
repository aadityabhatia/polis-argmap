import torch
import sys
import os
from guidance import models


def getTorchDeviceVersion():
    return f"""
    Device: {torch.cuda.get_device_name(0)}
    Python: {sys.version}
    PyTorch: {torch.__version__}
    CUDA: {torch.version.cuda}
    CUDNN: {torch.backends.cudnn.version()}
    """


def requireGPU():
    if not torch.cuda.is_available():
        raise Exception("No CUDA device found")


def getCUDAMemory():
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])
    total_memory = sum([torch.cuda.mem_get_info(i)[1]
                       for i in range(torch.cuda.device_count())])
    allocated_memory = sum([torch.cuda.memory_allocated(i)
                           for i in range(torch.cuda.device_count())])

    return free_memory, allocated_memory, total_memory


def printCUDAMemory(outputStream=sys.stdout):
    free_memory, allocated_memory, total_memory = getCUDAMemory()
    free_memory = round(free_memory/1024**3, 1)
    allocated_memory = round(allocated_memory/1024**3, 1)
    total_memory = round(total_memory/1024**3, 1)
    outputStream.write(
        f"CUDA Memory: {free_memory} GB free, {allocated_memory} GB allocated, {total_memory} GB total\n"
    )


def ensureCUDAMemory(required_memory_gb):
    required_memory = required_memory_gb * 1024**3
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])

    if free_memory >= required_memory:
        return True

    raise Exception(
        f"Insufficient CUDA memory: {round(free_memory/1024**3,1)} GB free, {required_memory_gb} GB required")


def loadLanguageModel(modelID, modelRevision=None, cudaMemoryGB=None, outputStream=sys.stdout):

    requireGPU()

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    if cudaMemoryGB is not None:
        ensureCUDAMemory(int(cudaMemoryGB))

    if not modelRevision:
        outputStream.write(f"Initializing language model: {modelID}...\n")
    else:
        outputStream.write(f"Initializing language model: {modelID} {modelRevision}...\n")

    languageModel = models.TransformersChat(modelID, device_map="auto", revision=modelRevision)

    outputStream.write("Language model initialized.\n")
    printCUDAMemory(outputStream)

    return languageModel
