# Daily Results Notion Monitor

The daily monitor summarizes completed experiment runs from the last 24 hours
and writes a Markdown/CSV report under:

```bash
/home/filip/server_code/results/daily_result_reports/
```

It scans `results/**/metrics.json`, so any experiment that follows the current
artifact layout is included. Seed directories touched in the same 24-hour window
but missing `metrics.json` are counted as incomplete.

## Run Once

```bash
cd /home/filip/server_code
./experiments/run_daily_results_notion_monitor.sh --once
```

## Daily tmux Monitor

The monitor is running in:

```bash
tmux attach -t main
# window: results_monitor
```

It wakes every day at `07:00` Europe/Paris by default. Useful overrides:

```bash
TIMEZONE=Europe/Paris RUN_HOUR=7 RUN_MINUTE=0 SINCE_HOURS=24 \
  ./experiments/run_daily_results_notion_monitor.sh --loop
```

## Notion Target

Reports target the Notion page:

```text
https://app.notion.com/p/3795e8e9c9f48103a58ce268951a58a2
```

The live Codex app connector can write to the page. A noninteractive
`codex exec` write from tmux currently fetches the page but has its Notion write
call cancelled by connector approval. Until an app automation hook or direct
Notion API token is available, the tmux monitor reliably generates local
reports and attempts the Notion append, but unattended Notion publishing is not
fully verified.
