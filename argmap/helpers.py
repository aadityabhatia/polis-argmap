import sys
import os
import datetime

embedModel = None
languageModel = None


def getTorchDeviceVersion():
    import torch
    return f"""\
Device: {torch.cuda.get_device_name(0)}
Python: {sys.version}
PyTorch: {torch.__version__}
CUDA: {torch.version.cuda}
CUDNN: {torch.backends.cudnn.version()}
"""


def printTorchDeviceVersion(outputStream=sys.stdout):
    outputStream.write(getTorchDeviceVersion())


def requireGPU():
    import torch
    if not torch.cuda.is_available():
        raise Exception("No CUDA device found")


def getCUDAMemory():
    import torch
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])
    total_memory = sum([torch.cuda.mem_get_info(i)[1]
                       for i in range(torch.cuda.device_count())])
    allocated_memory = sum([torch.cuda.memory_allocated(i)
                           for i in range(torch.cuda.device_count())])

    return free_memory, allocated_memory, total_memory


def printCUDAMemory(outputStream=sys.stderr):
    free_memory, allocated_memory, total_memory = getCUDAMemory()
    free_memory = round(free_memory/1024**3, 1)
    allocated_memory = round(allocated_memory/1024**3, 1)
    total_memory = round(total_memory/1024**3, 1)
    outputStream.write(
        f"CUDA Memory: {free_memory} GB free, {allocated_memory} GB allocated, {total_memory} GB total\n"
    )


def ensureCUDAMemory(required_memory_gb):
    import torch
    required_memory = required_memory_gb * 1024**3
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])

    if free_memory >= required_memory:
        return True

    raise Exception(
        f"Insufficient CUDA memory: {round(free_memory/1024**3,1)} GB free, {required_memory_gb} GB required")


def loadLanguageModel(outputStream=sys.stdout):

    global languageModel

    if languageModel is not None:
        return languageModel

    import torch
    from guidance import models

    CUDA_MINIMUM_MEMORY_GB = os.getenv("CUDA_MINIMUM_MEMORY_GB")
    MODEL_ID = os.getenv("MODEL_ID")
    MODEL_REVISION = os.getenv("MODEL_REVISION")

    if MODEL_ID is None:
        raise Exception(
            "Required: HuggingFace Model ID using MODEL_ID environment variable")

    requireGPU()

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    if CUDA_MINIMUM_MEMORY_GB is not None:
        ensureCUDAMemory(int(CUDA_MINIMUM_MEMORY_GB))

    outputStream.write(
        f"{datetime.datetime.now()} Initializing language model: {MODEL_ID}...\n")
    if MODEL_REVISION:
        outputStream.write(f"Model Revision: {MODEL_REVISION}\n")

    languageModel = models.TransformersChat(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="auto",
    )

    outputStream.write(
        f"{datetime.datetime.now()} Language model initialized.\n")
    printCUDAMemory(outputStream)

    return languageModel


def loadEmbeddingModel(outputStream=sys.stdout):
    import torch

    global embedModel

    if embedModel is not None:
        return embedModel


    EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")

    if EMBED_MODEL_ID is None:
        raise Exception(
            "Required: SentenceTransformer Model ID using EMBED_MODEL_ID environment variable")

    requireGPU()

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    outputStream.write(
        f"{datetime.datetime.now()} Initializing embedding model: {EMBED_MODEL_ID}...\n")

    from sentence_transformers import SentenceTransformer
    embedModel = SentenceTransformer(EMBED_MODEL_ID)

    outputStream.write(
        f"{datetime.datetime.now()} Embedding model initialized.\n")
    printCUDAMemory(outputStream)

    return embedModel


def unloadEmbeddingModel():
    import torch
    global embedModel
    embedModel = None
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
