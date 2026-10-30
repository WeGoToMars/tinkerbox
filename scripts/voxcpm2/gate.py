"""Practical FP16/content acceptance gate over saved VoxCPM2 validation results."""

import argparse
import json
from pathlib import Path


def numerical_ok(rows):
    for name, row in rows.items():
        decode = name.endswith('_decode')
        cosine = row.get('min_cosine', row.get('cosine', 0))
        relative = row.get('max_relative_l2', row.get('relative_l2', float('inf')))
        if not row.get('finite') or cosine < (0.90 if decode else 0.95) or relative > (0.50 if decode else 0.35):
            return False
    return bool(rows)


def character_error(actual, reference):
    actual = ''.join(c for c in actual.casefold() if c.isalnum())
    reference = ''.join(c for c in reference.casefold() if c.isalnum())
    if not reference:
        return 1.0
    previous = list(range(len(reference)+1))
    for i, a in enumerate(actual, 1):
        current = [i]
        for j, b in enumerate(reference, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]/len(reference)


def evaluate(report):
    report['gate'] = dict(prefill_min_cosine=0.95, prefill_max_relative_l2=0.35,
                        decode_min_cosine=0.90, decode_max_relative_l2=0.50,
                        max_asr_character_error=0.20, min_duration_ratio=0.5, max_duration_ratio=2.0)
    for result in report['cases'].values():
        gates = dict(numerical=numerical_ok(result['numerical']))
        if report.get('scope') != 'components':
            error = character_error(result.get('transcript', ''), result.get('upstream_transcript', ''))
            ratio = result.get('seconds', 0)/max(result.get('upstream_seconds', 0), 1e-8)
            result.update(asr_character_error=error, duration_ratio=ratio)
            gates.update(finite=result.get('finite', False), speech=error <= 0.20,
                         duration=0.5 <= ratio <= 2.0, stopped=not result.get('generation_error'))
        result['gates'] = gates
        result['passed'] = all(gates.values())
    report['passed'] = (set(report['cases']) == set(report.get('requested_cases', report['cases']))
                        and bool(report['cases']) and all(row['passed'] for row in report['cases'].values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = evaluate(json.loads(args.report.read_text()))
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    for name, result in report['cases'].items():
        print(name, result['gates'])
    print('PASS' if report['passed'] else 'FAIL')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
