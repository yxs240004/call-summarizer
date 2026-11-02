import dataclasses
import inspect

import trl
from trl import SFTConfig, SFTTrainer

print("trl", trl.__version__)
print("SFTConfig is dataclass:", dataclasses.is_dataclass(SFTConfig))
names = [f.name for f in dataclasses.fields(SFTConfig)] if dataclasses.is_dataclass(SFTConfig) else []
print("has assistant_only_loss:", "assistant_only_loss" in names)
print("has max_length:", "max_length" in names)
print("has max_seq_length:", "max_seq_length" in names)
print("has gradient_checkpointing:", "gradient_checkpointing" in names)
print("has chat_template_kwargs:", "chat_template_kwargs" in names)
print("has eval_strategy:", "eval_strategy" in names)
print()
print("=== SFTTrainer.__init__ ===")
print(inspect.signature(SFTTrainer.__init__))
print()
print("=== SFTConfig fields ===")
print(sorted(names))
