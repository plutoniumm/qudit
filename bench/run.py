import sys
import json
import numpy as np

sys.path.insert(0, "..")

import circuit as C
import noisy as N
import gd as G
import qec as Q


def run_circuit():
    results = {
        "meta": {"sizes": C.SIZES, "N": C.N, "warmup": C.WARMUP},
        "frameworks": {},
    }
    for name, fn in C.frameworks.items():
        means, stds = [], []
        for n in C.SIZES:
            try:
                t = fn(n)
                means.append(round(float(np.mean(t)), 4))
                stds.append(round(float(np.std(t)), 4))
            except Exception as e:
                means.append(None)
                stds.append(None)
                print(f"  [circuit/{name}/n={n}] ERR: {e}", file=sys.stderr)
        results["frameworks"][name] = {"mean_ms": means, "std_ms": stds}
        print(f"  circuit/{name}: done", file=sys.stderr)
    return results


def run_noisy():
    results = {}
    for key, spec in N.NOISE_TYPES.items():
        p = spec["p"]
        section = {
            "meta": {
                "sizes": N.SIZES,
                "N": N.N,
                "warmup": N.WARMUP,
                "label": spec["label"],
            },
            "frameworks": {},
        }
        for name, fn in spec["frameworks"].items():
            means, stds = [], []
            for wires in N.SIZES:
                try:
                    t = fn(wires, p)
                    means.append(round(float(np.mean(t)), 4))
                    stds.append(round(float(np.std(t)), 4))
                except Exception as e:
                    means.append(None)
                    stds.append(None)
                    print(f"  [noisy/{key}/{name}/n={wires}] ERR: {e}", file=sys.stderr)
            section["frameworks"][name] = {"mean_ms": means, "std_ms": stds}
            print(f"  noisy/{key}/{name}: done", file=sys.stderr)
        results[key] = section
    return results


def run_gd():
    labels = [cfg["label"] for cfg in G.CONFIGS]
    results = {
        "meta": {"configs": labels, "N": G.N, "warmup": G.WARMUP},
        "frameworks": {},
    }
    for name, fn in G.FRAMEWORKS.items():
        means, stds = [], []
        for cfg in G.CONFIGS:
            try:
                t = fn(cfg["wires"], cfg["layers"], cfg["steps"])
                means.append(round(float(np.mean(t)), 4))
                stds.append(round(float(np.std(t)), 4))
            except Exception as e:
                means.append(None)
                stds.append(None)
                print(f"  [gd/{name}/{cfg['label']}] ERR: {e}", file=sys.stderr)
        results["frameworks"][name] = {"mean_ms": means, "std_ms": stds}
        print(f"  gd/{name}: done", file=sys.stderr)
    return results


def run_qec():
    results = {
        "construct": {
            "meta": {"configs": Q.CONFIGS, "N": Q.N, "warmup": Q.WARMUP},
            "frameworks": {},
        },
        "apply": {
            "meta": {"configs": Q.CONFIGS, "N": Q.N, "warmup": Q.WARMUP},
            "frameworks": {},
        },
    }
    print("  qec/construct...", file=sys.stderr)
    results["construct"]["frameworks"] = Q.frameworks_construct()
    print("  qec/apply...", file=sys.stderr)
    results["apply"]["frameworks"] = Q.frameworks_apply()
    return results


BENCHES = {
    "circuit": run_circuit,
    "noisy": run_noisy,
    "gd": run_gd,
    "qec": run_qec,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="*", choices=list(BENCHES), help="run only these benchmarks"
    )
    parser.add_argument(
        "--out",
        default="../docs/public/bench.json",
        help="output path (default: ../docs/public/bench.json, use - for stdout)",
    )
    args = parser.parse_args()

    targets = args.only if args.only else list(BENCHES)
    output = {}

    for key in targets:
        print(f"running {key}...", file=sys.stderr)
        output[key] = BENCHES[key]()

    result = json.dumps(output, indent=2)

    if args.out == "-":
        print(result)
    else:
        with open(args.out, "w") as f:
            f.write(result)
        print(f"wrote {args.out}", file=sys.stderr)
