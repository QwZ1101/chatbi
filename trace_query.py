"""Run one real ChatBI request without starting a GUI. No credentials in CLI args."""
import argparse
import json
from chatbi_agent import init_agent_service
from trace_logging import sanitize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    args = parser.parse_args()
    bot = init_agent_service()
    final = []
    for final in bot.run([{'role': 'user', 'content': args.question}]):
        pass
    print(json.dumps(sanitize(final), ensure_ascii=False, indent=2))
