import torch

def assert_dicts_close(d1, d2, rtol=1e-5, atol=1e-8, ignore = {}):
    assert (d1.keys() - ignore) == (d2.keys() - ignore)
    for k in d1:
        if k in ignore:
            continue
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            torch.testing.assert_close(v1, v2, rtol=rtol, atol=atol)
        else:
            assert v1 == v2


def deep_update(base, overrides):
    """Recursively update a nested dict with overrides."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base