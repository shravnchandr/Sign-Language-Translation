"""Are a phrase model's confusions phonologically close sign pairs? ASL-LEX 2.0 coding (CC BY 4.0,
osf.io/zpha4: data/asl-lex/signdata.csv) of the true and the predicted sign of every single-sign error.

Labels are matched to ASL-LEX entries by name (exact or a numbered variant, e.g. hear_1 / hear_2) plus
a few non-signer synonyms (MANUAL, unreviewed); the rest stay unmapped. For a pair, the fewest of four
core parameters (handshape, minor location, movement, sign type) differing over the two signs' variant
entries is reported — the most similar reading, since which variant a dataset uses is unknown. The
baseline is the same measure over random pairs of mapped signs. A small difference says the forms are
close in ASL-LEX's coding, not that the dataset's recordings use those forms: a hypothesis for an
ASL reviewer, never a label correction.

  PYTHONPATH=research/models uv run python -m islr.evaluation.asl_lex_confusions \
      --per-clip popsign=results/evaluation/popsign-test/frozen07_dev.csv aslc=<…>_aslc_test.csv … \
      --out results/evaluation/frozen-phrases07-model-diagnostics/asllex_confusions.csv
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.data.vocab import canonical

# non-signer synonyms (unreviewed) for labels without a same-named entry; others left unmapped
MANUAL = {"dad": ["father"], "mom": ["mother"], "grandma": ["grandmother"], "grandpa": ["grandfather"],
          "thankyou": ["thank_you"], "tooth": ["teeth"], "shoe": ["shoes"], "police": ["policeman", "policeman_2"],
          "eye": ["eyes"], "garbage": ["trash"], "puzzle": ["jigsaw_puzzle"]}
CORE = {"handshape": "Handshape.2.0", "location": "MinorLocation.2.0", "movement": "Movement.2.0", "sign type": "SignType.2.0"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--asl-lex", type=Path, default=Path("data/asl-lex/signdata.csv"))
    p.add_argument("--per-clip", nargs="+", required=True, help="name=cross_dataset --per-clip CSV (ASL Citizen: identical glosses only)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lex = pd.read_csv(args.asl_lex, encoding="latin1")
    lex["id"] = lex.EntryID.str.lower().str.strip()
    lex = lex.set_index("id")
    vocab = canonical()
    lab = sorted(vocab, key=vocab.get)
    cand = {}
    for w in lab:
        c = [e for e in lex.index if e == w.lower() or re.fullmatch(re.escape(w.lower()) + r"_\d+", e)]
        cand[w] = c or MANUAL.get(w, [])

    def diffs(a, b):
        """fewest differing core parameters over the variant entries of two labels (None if unmapped)"""
        best = None
        for x, y in itertools.product(cand[a], cand[b]):
            n = sum(str(lex.at[x, c]) != str(lex.at[y, c]) for c in CORE.values())
            if best is None or n < best[0]:
                best = (n, x, y)
        return best

    fs = dict(x.split("=", 1) for x in args.per_clip)
    d = []
    for k, f in fs.items():
        x = pd.read_csv(f).assign(ds=k)
        if "kind" in x and x.source.eq("aslcitizen").any():
            x = x[x.kind == "identical"]
        d.append(x)
    d = pd.concat(d)
    err = d[(~d.correct.astype(bool)) & (d.n_tokens == 1)].copy()
    err["predsign"] = [lab[int(p)] for p in err.pred]
    err = err[err.predsign != err.sign]
    pairs = err.groupby(["sign", "predsign"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    pairs["d"] = [diffs(a, b) for a, b in zip(pairs.sign, pairs.predsign)]
    m = pairs[pairs.d.notna()].copy()
    m["k"] = m.d.str[0]
    mapped = [w for w in lab if cand[w]]
    print(f"labels mapped to ASL-LEX: {len(mapped)} / 250 (exact or numbered variant; {sum(w in MANUAL for w in mapped)} non-signer synonyms)")
    print(f"single-sign errors: {len(err)}; with both labels mapped: {m.n.sum()} ({m.n.sum() / len(err):.0%})")

    # baseline: for each true sign, every other mapped sign it could have been confused with, equally likely
    rng = np.random.default_rng(0)
    base = []
    for a in mapped:
        for b in rng.choice([w for w in mapped if w != a], 40, replace=False):
            base.append(diffs(a, b)[0])
    base = np.array(base)
    w = np.repeat(m.k.values, m.n.values)
    print("\ncore parameters differing (handshape, location, movement, sign type), error-weighted vs random pairs:")
    for k in range(5):
        print(f"  {k}: confusions {np.mean(w == k):5.1%}   random pairs {np.mean(base == k):5.1%}")
    print(f"  mean: confusions {w.mean():.2f}, random {base.mean():.2f}")

    # how many errors fall on a 'near' pair (≤ 1 core difference) vs random
    print(f"\nerrors on pairs differing in ≤ 1 core parameter: {np.mean(w <= 1):.0%} (random pairs {np.mean(base <= 1):.0%})")

    rows = []
    for _, r in m.head(40).iterrows():
        n, x, y = r.d
        diff = [f"{k}: {lex.at[x, c]} → {lex.at[y, c]}" for k, c in CORE.items() if str(lex.at[x, c]) != str(lex.at[y, c])]
        rows.append({"true": r.sign, "predicted": r.predsign, "errors": r.n, "entries": f"{x} / {y}", "core diff": n,
                     "differs in": "; ".join(diff) or "(none of the core four)"})
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print("\ntop confusions with both signs in ASL-LEX:")
    print(pd.DataFrame(rows).to_string(index=False))
    unm = pairs[pairs.d.isna()].head(15)
    print("\ntop confusions with a sign missing from ASL-LEX:", ", ".join(f"{a}→{b} ({n})" for a, b, n in zip(unm.sign, unm.predsign, unm.n)))
    pairs.assign(core_diff=pairs.d.str[0], entries=pairs.d.str[1] + " / " + pairs.d.str[2]).drop(columns="d").to_csv(
        args.out, index=False)


if __name__ == "__main__":
    main()
