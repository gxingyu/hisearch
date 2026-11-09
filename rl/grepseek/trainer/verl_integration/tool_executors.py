from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


@lru_cache(maxsize=None)
def get_tool_executor(tool_name: str, workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"grepseek-{tool_name}")
