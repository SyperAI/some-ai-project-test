import logging
import time
from functools import wraps
from typing import Callable


class Timer:
    def __init__(self, name: str, print_func: Callable = logging.debug):
        self.name = name
        self.print_func = print_func

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        self.print_func(f"{self.name} done in {elapsed:.2f}s.")

    @staticmethod
    def timer(print_func: Callable = logging.debug, name: str = None, kwargs_target: str = None,
              args_target: int = None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                target_naming = ""
                if kwargs_target is not None and kwargs.keys().__contains__(kwargs_target):
                    target_naming = f"of {kwargs[kwargs_target]}"
                elif args_target is not None and len(args) - 1 >= args_target:
                    target_naming = f"of {args[args_target]}"

                print_func(f"{name if name is not None else func.__name__} {target_naming} done in {elapsed:.2f}s.")

                return result

            return wrapper

        return decorator