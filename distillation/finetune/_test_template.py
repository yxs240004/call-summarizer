"""Verify how to force Qwen3 thinking OFF now that TRL dropped chat_template_kwargs."""
import subprocess

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
print("template has {% generation %}:", "{% generation %}" in t.chat_template)
print("template mentions enable_thinking:", "enable_thinking" in t.chat_template)

msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

print("\n--- default (enable_thinking undefined) ---")
print(repr(t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)[-90:]))

print("\n--- kwarg enable_thinking=False (works when calling directly) ---")
print(repr(t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)[-90:]))

t.chat_template = "{% set enable_thinking = false %}" + t.chat_template
print("\n--- after baking '{% set enable_thinking = false %}' prefix ---")
print(repr(t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)[-90:]))

print("\n--- full render with prefix (tail 200) ---")
print(t.apply_chat_template(
    [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
    tokenize=False, add_generation_prompt=True)[-200:])

print("\n=== does TRL still know the name chat_template_kwargs anywhere? ===")
print(subprocess.run(
    ["grep", "-rn", "chat_template_kwargs", "/usr/local/lib/python3.12/dist-packages/trl/"],
    capture_output=True, text=True).stdout[:1500] or "(no hits)")
