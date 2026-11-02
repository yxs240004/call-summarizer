"""Settle where the hand-rolled student template differs from the shipped one.

RESOLVED, and the answer was not what this file first suggested. probe_template_bytes.py
settled it by codepoint: the shipped template uses the 7- and 8-character markers that
tokenize to the added tokens 248068 / 248069, while verify_student.py's hand-rolled
constant spelled them as the literal words ' think' / ' response' (tokens 7047 / 1965).
Same defect class as the Qwen3-8B episode. The constant is now marked superseded and
train_student.py derives its labels from the shipped template instead, so this
comparison has nothing left to guard.

Kept for the record: the byte-and-token-id method below is the one that works.

verify_student.py reported MATCH: False at char 137 while both rendered strings
printed identically, which is the exact trap from the Qwen3-8B work: a difference
that `repr` makes invisible. So this compares bytes and, more to the point, the
token ids each render produces -- because identical token ids are the only thing
that actually guarantees training and inference agree.
"""
from verify_student import PROBE, STUDENT_TEMPLATE, flat_template

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"


def render(tok, template, messages, add_generation_prompt):
    tok.chat_template = template
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    shipped = flat_template(tok)

    want = render(tok, shipped, PROBE, False)
    got = render(tok, STUDENT_TEMPLATE, PROBE, False)

    print(f"len(shipped)={len(want)}  len(hand-rolled)={len(got)}")
    print(f"equal        = {want == got}")

    if want != got:
        idx = next(i for i, (a, b) in enumerate(zip(want, got)) if a != b)
        lo, hi = max(0, idx - 8), idx + 8
        print(f"\nfirst differing index: {idx}")
        print(f"  shipped     {want[lo:hi].encode('utf-8').hex(' ')}")
        print(f"  hand-rolled {got[lo:hi].encode('utf-8').hex(' ')}")
        print(f"  shipped     {want[lo:hi]!r}")
        print(f"  hand-rolled {got[lo:hi]!r}")
        print(f"  length delta: shipped {len(want) - len(got):+d} chars")
        # Where do the two strings re-converge, if they do?
        tail = 0
        while tail < min(len(want), len(got)) and want[-1 - tail] == got[-1 - tail]:
            tail += 1
        print(f"  identical trailing chars: {tail}")
        print(f"  shipped     tail {want[len(want) - tail - 12:]!r}")
        print(f"  hand-rolled tail {got[len(got) - tail - 12:]!r}")

    print("\n-- token ids each render produces --")
    tok.chat_template = shipped
    ids_shipped = tok(want, add_special_tokens=False)["input_ids"]
    tok.chat_template = STUDENT_TEMPLATE
    ids_hand = tok(got, add_special_tokens=False)["input_ids"]

    print(f"  shipped      {len(ids_shipped):>4} ids, tail {ids_shipped[-12:]}")
    print(f"  hand-rolled  {len(ids_hand):>4} ids, tail {ids_hand[-12:]}")
    print(f"  IDENTICAL    : {ids_shipped == ids_hand}")

    if ids_shipped != ids_hand:
        j = next(i for i, (a, b) in enumerate(zip(ids_shipped, ids_hand)) if a != b)
        print(f"  first differing token index {j}:")
        print(f"    shipped     {ids_shipped[j]}:{ascii(tok.convert_ids_to_tokens(ids_shipped[j]))}")
        print(f"    hand-rolled {ids_hand[j]}:{ascii(tok.convert_ids_to_tokens(ids_hand[j]))}")


if __name__ == "__main__":
    main()
