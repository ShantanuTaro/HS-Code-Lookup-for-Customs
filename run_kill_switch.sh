#!/bin/sh
# Wait for the CROSS crawl to finish, then run the 500-case kill-switch backtest.
while pgrep -f "cross.py --from-year" > /dev/null; do sleep 30; done
echo "crawl done: $(wc -l < data/rulings.jsonl) rulings"
exec .venv/bin/python backtest.py --cases 500 --workers 2
