"""Export one existing trace; never fabricate missing events."""
import argparse
import json
from pathlib import Path
from trace_logging import sanitize


def export_trace(source, destination):
    events = [json.loads(line) for line in Path(source).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not events or events[0]['event'] != 'request_start':
        raise ValueError('Missing request_start')
    ids = {e['trace_id'] for e in events}
    if len(ids) != 1 or [e['seq'] for e in events] != list(range(1, len(events) + 1)):
        raise ValueError('Mixed trace IDs or missing events')
    end = events[-1]
    starts = {e['span_id'] for e in events if e['event'] == 'call_start'}
    stops = {e['span_id'] for e in events if e['event'] in ('call_return', 'call_failed')}
    truncated = any(e['data'].get('truncated') for e in events)
    report = {'trace_id': events[0]['trace_id'], 'mode': events[0]['mode'],
              'complete_record': end['event'] == 'request_end' and starts == stops and not truncated,
              'warning': 'New execution record. offline_test uses test doubles; completed does not mean answer correct. Review business data before sharing.',
              'events': sanitize(events)}
    Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='One logs/traces/YYYY-MM-DD/<trace_id>.jsonl file')
    parser.add_argument('--out', required=True, help='Destination JSON, manually review before sharing')
    args = parser.parse_args()
    report = export_trace(args.source, args.out)
    print(json.dumps({k: report[k] for k in ('trace_id', 'mode', 'complete_record')}, ensure_ascii=False))
