"""Relax only the front/back metadata gate for one frontal hair photograph.

Tensor packing, reference order, time positions, model and sampler are untouched.
The exact modified function is saved for review and hashed in generation records.
"""
import inspect
from common import *

def install():
    import native_reference_strategy as strategy
    original=inspect.getsource(strategy.reference_segments)
    old='if kind in {"hair", "garment", "outfit"} and representation is None and not official_garment:'
    new=old[:-1]+' and not (kind == "hair" and len(batch["references"]) == 1 and scalar(item.get("view", "")) == "front"):'
    assert original.count(old)==1,'Upstream validation changed; re-audit required'
    modified=original.replace(old,new)
    exec(compile(modified,'a849_single_front_hair_metadata_adapter','exec'),strategy.__dict__)
    # A shard-specific receipt avoids concurrent writes to one audit file.
    return dict(upstream_sha256=sha(NATIVE/'scripts/common/native_reference_strategy.py'),adapter_sha256=sha(__file__),modified_function=modified,scope='Only single front hair metadata validation. Native v7 reference tokens/positions unchanged; original hair type retained.')
