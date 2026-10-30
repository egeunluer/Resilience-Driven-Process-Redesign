#!/usr/bin/env python3
"""
Mine XOR branching weights for the RIMS simulator from the real event log.

event_trace.Token.next_step() chooses among the transitions enabled in a case's
current marking via define_xor_next_activity(), which reads the input JSON's
"probability" section, keyed by transition label (or by transition name for
silent transitions). Every input_*.json shipped in example/ had that section
empty, so every decision point was chosen uniformly at random -- e.g. in
Consulta, ~27% of simulated cases went through 'Recepcion de documentos'
against 2.4% in the real log.

This script token-replays the real log on the dataset's own Petri net, counts
how often each transition fired (silent ones included), and writes those counts
into "probability" as relative weights. define_xor_next_activity() normalizes
them over whichever transitions are enabled, so at an XOR split each branch is
taken with its observed conditional frequency. A transition never fired during
replay gets weight 0 (never chosen while an observed alternative is enabled).

Usage:
  python3 mine_branch_probabilities.py --dataset ConsultaDataMining201618 \\
      --inputs input_ConsultaDataMining201618.json input_ConsultaDataMining20161820.json
"""
import argparse
import json
import os
import re
from collections import Counter

import pm4py


def mine_weights(log_path, pnml_path):
    log = pm4py.read_xes(log_path)
    net, im, fm = pm4py.read_pnml(pnml_path)
    replay = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)
    fired = Counter(t for trace in replay for t in trace['activated_transitions'])
    fit = sum(trace['trace_is_fit'] for trace in replay) / len(replay)

    labels = Counter(t.label for t in net.transitions if t.label)
    duplicated = sorted(l for l, n in labels.items() if n > 1)
    if duplicated:
        # The simulator keys visible transitions by label, so duplicates share one
        # weight: their counts are summed. Reported so this is never silent.
        print(f"WARNING: duplicate transition labels share one weight: {duplicated}")

    weights = Counter()
    for t in net.transitions:
        weights[t.label if t.label else t.name] += fired[t]
    return dict(weights), fit, len(replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help="Folder name under example/")
    parser.add_argument('--inputs', nargs='+', required=True,
                        help="input_*.json file names (in example/<dataset>/) to write the weights into")
    args = parser.parse_args()

    folder = os.path.join('example', args.dataset)
    weights, fit, n_traces = mine_weights(os.path.join(folder, f'{args.dataset}.xes'),
                                          os.path.join(folder, f'{args.dataset}.pnml'))
    print(f"Token replay: {n_traces} traces, {fit:.1%} fit perfectly; "
          f"{sum(1 for w in weights.values() if w == 0)} of {len(weights)} transitions never fired")

    source = (
        f"Transition firing counts from token-based replay of {args.dataset}.xes on "
        f"{args.dataset}.pnml ({n_traces} traces, {fit:.1%} perfectly fitting), written by "
        f"mine_branch_probabilities.py. Relative weights, normalized over the enabled "
        f"transitions at each decision."
    )
    block = ('"probability": {\n'
             + ',\n'.join(f'        {json.dumps(k)}: {v}' for k, v in sorted(weights.items()))
             + f'\n    }},\n    "probability_source": {json.dumps(source)}')
    for name in args.inputs:
        path = os.path.join(folder, name)
        with open(path) as f:
            text = f.read()
        # The input files are hand-formatted (one resource per line), so only the
        # "probability" block is replaced in the text -- re-dumping the whole JSON
        # would reformat thousands of unrelated lines. A previous run's
        # "probability_source" line is dropped first, so reruns stay idempotent.
        text = re.sub(r',\s*"probability_source":\s*"(?:[^"\\]|\\.)*"', '', text)
        text, n = re.subn(r'"probability":\s*\{[^{}]*\}', lambda m: block, text)
        if n != 1:
            raise ValueError(f"Expected exactly one flat \"probability\" block in {path}, found {n}")
        json.loads(text)   # still valid JSON
        with open(path, 'w') as f:
            f.write(text)
        print(f"Wrote {len(weights)} branch weights to {path}")


if __name__ == '__main__':
    main()
