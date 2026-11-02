"""Settle, in bytes only, what the student's shipped template actually contains.

diff_templates.py showed the shipped and hand-rolled student templates produce the
same 45 tokens except one: shipped gives the special token 248068, hand-rolled gives
the text token 7047. Reading the two renders as `repr` did not explain it, and repr
output in a terminal has already fooled me once on this exact question (the
7274/151667 episode on the 8B). So every answer below is a hex dump, an ordinal, or
a boolean -- nothing that a terminal can silently rewrite.

Second reason for the ordinals: the needles below are BUILT with chr() rather than
written as literals. If a tool-call round trip ever mangles a marker, the ordinals
printed here will disagree with the ones this docstring claims, and that shows up
immediately instead of silently probing for the wrong string.
"""
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"

LT = chr(0x3C)   # '<'
GT = chr(0x3E)   # '>'
SLASH = chr(0x2F)

CANDIDATES = {
    "angle-open": LT + "thinking" + GT,
    "angle-close": LT + SLASH + "thinking" + GT,
    "space-open": " thinking",
    "space-close": " response",
    "angle-resp-open": LT + "response" + GT,
    "angle-resp-close": LT + SLASH + "response" + GT,
    "space-resp-close": " response",
}


def hx(s):
    return s.encode("utf-8").hex(" ")


def ords(s):
    return " ".join(str(ord(c)) for c in s)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    template = tok.chat_template

    print(f"chat_template type : {type(template).__name__}")
    print(f"chat_template len  : {len(template)}")

    print("\n-- what each needle actually is, by ordinal (self-check) --")
    for name, cand in CANDIDATES.items():
        print(f"  {name:<18} len={len(cand):<3} ords={ords(cand)}  hex={hx(cand)}")

    print("\n-- needle -> token ids (which marker is a single special token?) --")
    for name, cand in CANDIDATES.items():
        ids = tok(cand, add_special_tokens=False)["input_ids"]
        print(f"  {name:<18} -> {ids}")

    print("\n-- is each needle present in the shipped template? --")
    for name, cand in CANDIDATES.items():
        print(f"  {name:<18} in template: {cand in template}")

    print("\n-- added_tokens_decoder, ids 248044..248075 --")
    decoder = tok.added_tokens_decoder
    for token_id in sorted(decoder):
        if 248044 <= token_id <= 248075:
            entry = decoder[token_id]
            print(f"  {token_id}: hex={hx(entry.content):<48} ords={ords(entry.content):<24} special={entry.special}")

    print("\n-- every angle-bracket run in the template, as hex --")
    # Find each '<' and print the hex up to and including the matching '>'.
    found = 0
    i = template.find(LT)
    while i != -1 and found < 40:
        j = template.find(GT, i)
        if j == -1:
            break
        run = template[i:j + 1]
        if len(run) <= 24:
            print(f"  at {i:>5}: {hx(run)}")
            found += 1
        i = template.find(LT, j + 1)
    print(f"  ({found} runs)")

    print("\n-- booleans that decide the fix --")
    open_needle = CANDIDATES["angle-open"]
    space_needle = CANDIDATES["space-open"]
    print(f"  template contains angle-open  : {open_needle in template}")
    print(f"  template contains space-open  : {space_needle in template}")
    print(f"  angle-open tokenizes to [248068]: "
          f"{tok(open_needle, add_special_tokens=False)['input_ids'] == [248068]}")
    print(f"  space-open tokenizes to [7047]  : "
          f"{tok(space_needle, add_special_tokens=False)['input_ids'] == [7047]}")


if __name__ == "__main__":
    main()
