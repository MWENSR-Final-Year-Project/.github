# Final Year Project

Research implementation of an **AlphaZero-inspired agent** for the game of **Dots and Boxes**, trained entirely through self-play on Google Kubernetes Engine.

> *To what extent can an AlphaZero-inspired agent, trained solely through self-play, independently discover and apply strategic principles traditionally used by human players in Dots and Boxes?*

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      GKE cluster                        │
│                                                         │
│  dot-zero-selfplay   ──games──►   Bigtable              │
│  (CPU pool, 2–32     ◄─weights─   (game_queue)          │
│   pods, HPA)                          │                 │
│                                       │ positions       │
│                                       ▼                 │
│  dot-zero-train  ──checkpoint──►    GCS                 │
│  (CPU pool, 1 pod)  ◄─best model─  (models/             │
│                                     checkpoints/        │
│                                     eval-requests/)     │
│                                       ▲                 │
│  dot-zero-eval   ──eval request───────┘                 │
│  (CPU pool, 1–8  ──metrics──► Bigtable + GCS JSONL      │
│   pods, HPA)                                            │
└─────────────────────────────────────────────────────────┘
```

Self-play workers generate games → trainer consumes positions from Bigtable, trains `DotZeroNet`, publishes weights to GCS → workers hot-swap the new model → trainer writes eval requests → eval workers score each checkpoint. The loop repeats.

---

## Repositories

| Repository | Description |
|---|---|
| [dot-zero-game](https://github.com/MWENSR-Final-Year-Project/dot-zero-game) | Game engine and agent library. Dots & Boxes rules, board/state/move representation, action encoding for RL, and six agents: `Random`, `Greedy`, `Defensive`, `Heuristic`, `MCTSAgent` (pure UCT), `NGMCTSAgent` (PUCT + neural guidance). |
| [dot-zero-model](https://github.com/MWENSR-Final-Year-Project/dot-zero-model) | Neural network and search. `DotZeroNet` — a dual-head residual CNN (policy + value) built in TensorFlow/Keras. PUCT search with Dirichlet noise, temperature scheduling, and 8-fold symmetry augmentation. Designed for TPU training on GKE. |
| [dot-zero-selfplay](https://github.com/MWENSR-Final-Year-Project/dot-zero-selfplay) | Self-play worker service. CPU pods poll GCS for the latest model checkpoint, run PUCT games in parallel, and stream positions to Bigtable. Hot-swaps the model automatically when the trainer publishes a new one. Scales 2–32 replicas via HPA. |
| [dot-zero-train](https://github.com/MWENSR-Final-Year-Project/dot-zero-train) | Training service. Reads positions from Bigtable into a replay buffer, runs gradient steps (policy + value loss), optionally gates the new model in an arena match before promotion, and publishes the best model to GCS. Logs to TensorBoard and Bigtable. |
| [dot-zero-eval](https://github.com/MWENSR-Final-Year-Project/dot-zero-eval) | Evaluation service. Polls GCS for eval request markers written by the trainer, runs the full evaluation suite — arena matches vs all baselines, endgame accuracy tests (double-cross, parity sacrifice, loop control), Elo rating updates — and writes results to Bigtable and GCS JSONL. |
| [dot-zero-infra](https://github.com/MWENSR-Final-Year-Project/dot-zero-infra) | Infrastructure. Terraform provisions the GCS bucket, Bigtable instance, GKE cluster, Artifact Registry, and Workload Identity bindings. Helm charts deploy all three services with shared ConfigMaps and per-run value overrides. Docker Compose runs the full pipeline locally without GCP. |
| [dot-zero](https://github.com/MWENSR-Final-Year-Project/dot-zero) | Original monorepo. The initial unified implementation — game engine, MCTS, neural network, training loop, and baseline agents in a single package. Superseded by the service split above. |

---

## Model

`DotZeroNet` is a convolutional residual network with two output heads:

```
Input (n+1, n+1, 4)   — horizontal edges · vertical edges · player boxes · opponent boxes
    ↓
Initial Conv + BatchNorm + ReLU
    ↓
Residual Tower  (6 × ResidualBlock, 64 channels)
    ↓
    ├── Policy Head  →  legal-masked logits over all edge placements
    └── Value Head   →  scalar win probability ∈ [−1, 1]
```

Search uses **PUCT** (AlphaZero's MCTS variant). During self-play, Dirichlet noise (`α = 10 / action_size`, `ε = 0.25`) is injected at the root and temperature is held at 1.0 for the first ~30% of moves, then dropped to 0.

---

## Evaluation

Each checkpoint is scored against five baseline agents (Random, Greedy, Defensive, Heuristic, Pure MCTS) and across four endgame categories:

| Category | What is tested |
|---|---|
| Capture forced | Takes a free box immediately |
| Double-cross | Sacrifices 2 boxes to force opponent to open the next chain |
| Parity sacrifice | Gives up a chain to control parity |
| Loop control | Handles 2×2 loop timing correctly |

Elo ratings are tracked across all agents for every training iteration.

---

## Stack

- **Language:** Python 3.12
- **Neural network:** TensorFlow / Keras (TPU-compatible, channels-last)
- **Game logic:** NumPy, Pydantic
- **Cloud:** Google Cloud (GKE, GCS, Bigtable, Workload Identity)
- **Infrastructure:** Terraform, Helm, Docker
- **Container registry:** GitHub Container Registry (`ghcr.io/mwensr-final-year-project`)

---

**Author:** Jason Kitamirike — BSc Computer Science Final Year Project
