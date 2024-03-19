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
CUDNN: {torch.backends.cudnn.version()}"""


def printTorchDeviceVersion():
    print(getTorchDeviceVersion())


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


def printCUDAMemory():
    free_memory, allocated_memory, total_memory = getCUDAMemory()
    free_memory = round(free_memory/1024**3, 1)
    allocated_memory = round(allocated_memory/1024**3, 1)
    total_memory = round(total_memory/1024**3, 1)
    print(f"CUDA Memory: {free_memory} GB free, {allocated_memory} GB allocated, {total_memory} GB total",
          flush=True, file=sys.stderr)


def ensureCUDAMemory(required_memory_gb):
    import torch
    required_memory = required_memory_gb * 1024**3
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])

    if free_memory >= required_memory:
        return True

    raise Exception(
        f"Insufficient CUDA memory: {round(free_memory/1024**3,1)} GB free, {required_memory_gb} GB required")


def loadLanguageModel():

    global languageModel

    if languageModel is not None:
        return languageModel

    import torch
    from guidance import models

    MODEL_ID = os.getenv("MODEL_ID")
    MODEL_REVISION = os.getenv("MODEL_REVISION")

    if MODEL_ID is None:
        raise Exception(
            "Required: HuggingFace Model ID using MODEL_ID environment variable")

    requireGPU()

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    MODEL_MINIMUM_MEMORY_GB = os.getenv("MODEL_MINIMUM_MEMORY_GB")
    if MODEL_MINIMUM_MEMORY_GB is not None:
        ensureCUDAMemory(int(MODEL_MINIMUM_MEMORY_GB))

    print(f"{datetime.datetime.now()} Initializing language model: {MODEL_ID}...")
    if MODEL_REVISION:
        print(f"Model Revision: {MODEL_REVISION}")

    languageModel = models.TransformersChat(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="auto",
    )

    print(f"{datetime.datetime.now()} Language model initialized.")
    printCUDAMemory()

    return languageModel


def loadEmbeddingModel():
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

    EMBED_MODEL_MINIMUM_MEMORY_GB = os.getenv("EMBED_MODEL_MINIMUM_MEMORY_GB")
    if EMBED_MODEL_MINIMUM_MEMORY_GB is not None:
        ensureCUDAMemory(int(EMBED_MODEL_MINIMUM_MEMORY_GB))

    print(f"{datetime.datetime.now()} Initializing embedding model: {EMBED_MODEL_ID}...")

    from sentence_transformers import SentenceTransformer
    embedModel = SentenceTransformer(EMBED_MODEL_ID)

    print(f"{datetime.datetime.now()} Embedding model initialized.")
    printCUDAMemory()

    return embedModel


def unloadEmbeddingModel():
    import torch
    global embedModel
    embedModel = None
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
