"""Compatibility helper imported by the paper MDN trainer."""


def normalize_state_dict(state_dict):
    out = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ("module.", "_orig_mod."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = value
    return out
