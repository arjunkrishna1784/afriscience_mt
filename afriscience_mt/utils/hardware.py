import gc
import torch
import time
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

def cleanup_gpu(client=None, logger=None):
    """
    Corrected: Now accepts client and logger to prevent TypeError.
    """
    if logger:
        logger.info("Releasing GPU resources and destroying vLLM engine...")
    
    # 1. vLLM Specific Cleanup
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        if logger:
            logger.debug(f"vLLM cleanup notice: {e}")

    # 2. Object cleanup
    if client:
        # Explicitly delete the vLLM engine if it exists
        if hasattr(client, "llm"):
            del client.llm
        del client

    # 3. Standard PyTorch and Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Wait for background processes to settle
    time.sleep(2)