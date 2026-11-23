# CommunityBridge AI

Turns one plain-language goal, such as *"I need to get from Ceduna to my cardiology appointment in Adelaide next month"*, into a single support plan for Aboriginal and Torres Strait Islander people in country South Australia who are travelling to Adelaide specialists. Every factual claim in the plan cites its source, and nothing happens until the person approves it.

> **Prototype notice**
> - All data is **synthetic**. No real people or health details are used.
> - Transport, accommodation and calendar services are **simulated** and labelled as such.
> - Partner organisations are prospective only. No community endorsement is claimed.
> - It gives no clinical advice and makes no real bookings, payments or PATS claims.

## How it works
Six agents (Coordinator, Service Discovery, Eligibility, Logistics, Verifier, Consent) run in a LangGraph state graph. A language model only reads the goal, coordinates the agents and explains results. Deterministic tools decide eligibility, logistics costs and consent. PATS eligibility comes from a rules engine whose rules are versioned data (`data/rules/pats_v26_4.yaml`) encoded from the SA Health *PATS Guidelines for Assessment* V26.4, and every rule cites its guideline section. The guideline PDF is not included here.

## Setup (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
```
Requires Python 3.11 or later.

## Language model (Ollama)
The agents use a local model through [Ollama](https://ollama.com) to read the goal and word explanations.
```powershell
ollama pull <a model you have room for>      # developed with qwen2.5:7b
# put the same name in .env:  OLLAMA_MODEL=qwen2.5:7b
python -m communitybridge.llm.check          # prints a reply if the model is reachable
```

## Run it
**In the browser** (each approval gate is a web form):
```powershell
# optional, in .env: COMMUNITYBRIDGE_TODAY=2026-10-11
uvicorn communitybridge.web.app:app
```
Open http://127.0.0.1:8000 and type a goal, for example *"I need to get from Ceduna to my cardiology appointment in Adelaide next month. I'm Aboriginal and my sister will come with me."* The synthetic data is built around November 2026, so set `COMMUNITYBRIDGE_TODAY=2026-10-11` for "next month" to mean November. The first request builds the SQLite database from `data/` by itself.

**In the terminal** (a person answers every gate by typing; nothing approves itself):
```powershell
python -m communitybridge.cli ceduna          # uses the Ollama model
python -m communitybridge.cli ceduna --fake   # no model needed: replays the scenario's expected reading
```

## Layout
- `src/communitybridge/`: agents, graph, deterministic tools, simulated services, web app
- `data/`: PATS rules, synthetic services and distances, simulated fixtures, consent policy
- `scripts/seed_db.py`: builds the SQLite database from `data/`
