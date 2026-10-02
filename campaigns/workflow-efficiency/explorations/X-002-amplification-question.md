# X-002 — Research question first

Filip accepted: Does bidirectional amplification improve early learning in a tiny resistive network under matched initialization, data, and optimizer settings?

Reasoning: amplification can alter gradient propagation and hence early learning at fixed raw learning rates. It can also alter loss scale or dynamics, so report absolute losses alongside relative improvement. A tiny paired run can exercise hypothesis → decision → implementation → monitoring → review, but cannot establish the general paper claim. Existing runner exposes initialization and per-epoch data-order hashes; reuse those instead of adding separate checkpoint replays.

Earlier mistake: exp-001 selected a one-arm config and delegated Monitor before this scientific question was agreed. It was superseded without execution. No new broad qualification is needed under the September25 policy.
