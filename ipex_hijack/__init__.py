import functools


def log(msg):
    print(f"[ipex_enhance] {msg}")


def hijack_message(msg=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if msg:
                log(msg)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def asfp16(v):
    return v.to(torch.half) if v is not None else None


def astype(v, type):
    return v.to(type) if v is not None else None