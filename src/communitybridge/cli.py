"""Run a scenario end to end from the terminal: `python -m communitybridge.cli ceduna`.

Serves all six agents. It builds the same graph the web app will use (`graph/build.py`), with the
real tools, the real PATS rules and the SIMULATED services, and puts a person at the four approval
gates by asking for typed answers. At the end it prints the plan, the SIMULATED requests and the
whole agent trace. This is the demo path: one complete multi-step user scenario with approval gates and a
visible agent trace. `ceduna` is the core journey, scenario SC-01.

Design choices, and why:
- **A person answers every gate.** There is no "approve everything" switch. A claim of zero
  unapproved actions (success target) means nothing if the demo can approve for itself. An empty
  or unclear line asks again, and if the keyboard goes away (end of input) the run stops with
  nothing approved.
- **"Next month" means the scenario's own month.** SC-01 says "next month" and its fixtures are
  built around November 2026 (the December ones are built to fail, for re-planning:). Run on
  3 November, a live clock would say December and the core journey would re-plan. So the date the
  Coordinator resolves relative dates against is the scenario file's `reference_date`, printed at
  the top, and `--today` changes it. The Verifier is different: it judges how old a *source* is, so
  it uses the real date.
- **`--fake` is for tests and offline runs.** It replays the scenario's expected extraction with the
  FakeProvider and says so in the first lines.
  Without it, the model named by `OLLAMA_MODEL` is used.
- **Each run is a new journey.** The thread id is random, so a second run never resumes the first.
  The checkpoint file keeps a stopped run, but resuming from the CLI is not built.
- **The contact details are made up.** Gate 4 needs a name, phone number and date of birth to show,
  so `DEMO_CONTACT` holds a synthetic person.
"""

import argparse
import json
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from langgraph.types import Command

from communitybridge.agents.coordinator import GATE_ONE, GATE_TWO
from communitybridge.agents.logistics import GATE_THREE
from communitybridge.cli_text import (
    ACTION_CHOICES,
    DETAILS_CHOICES,
    PLAN_CHOICES,
    choose_from,
    gate_three_choice,
    render_actions,
    render_gate,
    render_outcome,
    render_plan,
    render_trace,
)
from communitybridge.config import CHECKPOINT_DB_PATH, DB_PATH, ConfigError, ollama_settings
from communitybridge.db.seed import seed_database
from communitybridge.graph.build import build_graph, default_tools
from communitybridge.graph.checkpoint import open_checkpointer
from communitybridge.graph.state import GraphState
from communitybridge.llm.base import LLMProvider, LLMUnavailableError
from communitybridge.llm.fake import FakeProvider
from communitybridge.llm.ollama import OllamaProvider
from communitybridge.models.contact import ClientContact
from communitybridge.models.scenario import Scenario
from communitybridge.scenarios import load_scenarios
from communitybridge.tools.vocabulary import load_vocabulary

# The command-line names for the scenarios a person can run. Only the core journey so far.
SCENARIOS = {"ceduna": "SC-01"}

# A made-up person for Gate 4 to show. Nothing here is, or looks like, a real record.
DEMO_CONTACT = ClientContact(
    synthetic=True,
    given_name="Alex",
    family_name="Synthetic",
    contact_phone="0400 000 000",
    date_of_birth=date(1980, 1, 2),
    mobility_assistance_needed=False,
)


class NoAnswerError(Exception):
    """The keyboard went away (end of input) while a gate was waiting. Nothing is approved."""

    def __init__(self, gate: str) -> None:
        super().__init__(gate)
        self.gate = gate


@dataclass(frozen=True)
class CliResult:
    """How a run ended: the exit code, and the final state (None if it never started)."""

    exit_code: int
    state: GraphState | None


class Console:
    """Asks the person at the keyboard to answer a gate, and only ever returns what they typed.

    `input_fn` is `input` in the real program and a scripted typist in tests. Nothing here has a
    default answer: a line that is not one of the accepted words is asked for again.
    """

    def __init__(self, input_fn: Callable[[str], str], out: TextIO) -> None:
        self._input = input_fn
        self._out = out
        self._gate = ""

    def ask(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Show a gate and return the answer to hand back to `interrupt()`."""
        self._gate = str(payload["gate"])
        self._out.write("\n" + render_gate(payload))
        if self._gate == GATE_ONE:
            return self._gate_one()
        if self._gate == GATE_TWO:
            return {"decision": self._choose("[a]pprove the plan or [r]eject it > ", PLAN_CHOICES)}
        if self._gate == GATE_THREE:
            return self._gate_three(len(payload["options"]))
        # `render_gate` above refuses any gate it has no screen for, so this is the fourth.
        return self._gate_four(payload["actions"])

    def _read(self, prompt: str) -> str:
        try:
            return self._input(prompt)
        except EOFError:
            raise NoAnswerError(self._gate) from None

    def _choose(self, prompt: str, choices: Mapping[str, str]) -> str:
        while True:
            choice = choose_from(choices, self._read(prompt))
            if choice is not None:
                return choice
            words = ", ".join(sorted(choices))
            self._out.write(f"Please type one of: {words}. An empty line is not an answer.\n")

    def _gate_one(self) -> dict[str, Any]:
        while True:
            prompt = "[c]onfirm, [e]dit your wording or [d]ecline > "
            choice = self._choose(prompt, DETAILS_CHOICES)
            if choice != "edit":
                return {"decision": choice}
            wording = self._read("Type your request again > ").strip()
            if wording:
                return {"decision": "edit", "goal_text": wording}
            self._out.write("Please type your request, or choose another answer.\n")

    def _gate_three(self, option_count: int) -> dict[str, Any]:
        prompt = f"Type an option number (1-{option_count}) to approve it, or [d]ecline > "
        while True:
            answer = gate_three_choice(self._read(prompt), option_count)
            if answer is not None:
                return answer
            self._out.write(
                f"Please type a number from 1 to {option_count}, or d. "
                "An empty line is not an answer.\n"
            )

    def _gate_four(self, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        decisions = {}
        for number, action in enumerate(actions, start=1):
            prompt = (
                f"Action {number} of {len(actions)} ({action['action']}): [a]pprove or [d]ecline > "
            )
            decisions[action["action"]] = self._choose(prompt, ACTION_CHOICES)
        return {"decisions": decisions}


def _provider(
    fake: bool, scenario: Scenario, environ: Mapping[str, str] | None
) -> tuple[LLMProvider, str]:
    """The model behind the Coordinator and the Eligibility Agent, and a line saying which."""
    if fake:
        fields = scenario.expected_fields.model_dump(mode="json", exclude={"confirmed_by_user"})
        # The second call (the plain-language explanation) fails on purpose, so the Eligibility
        # Agent uses its checked template and the whole run needs no model.
        provider = FakeProvider([json.dumps(fields), LLMUnavailableError("--fake: no model")])
        return provider, (
            "No language model was used (--fake): the scenario's expected details stand in for "
            "the Coordinator's reading, and the explanation comes from the checked template."
        )
    settings = ollama_settings(environ)
    return OllamaProvider(settings), f"Language model: Ollama, {settings.model}"


def _header(scenario: Scenario, reference: date, scenario_name: str, model_line: str) -> str:
    return (
        f"CommunityBridge AI: {scenario_name} demo\n"
        f"Scenario {scenario.id}: {scenario.title}\n"
        f'Request: "{scenario.goal}"\n'
        f'Date used to resolve "next month": {reference.isoformat()} '
        "(the scenario's own date; change it with --today)\n"
        f"{model_line}\n"
        "Everything here is synthetic. Booking requests and data releases go to SIMULATED "
        "services only: nothing real is booked, paid for or sent.\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m communitybridge.cli",
        description="Run a CommunityBridge AI scenario end to end, answering the approval gates "
        "yourself.",
    )
    parser.add_argument("scenario", choices=sorted(SCENARIOS), help="the scenario to run")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use no language model: replay the scenario's expected details (tests, offline runs)",
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="resolve relative dates such as 'next month' against this date "
        "(default: the scenario's own reference date)",
    )
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    out: TextIO | None = None,
    err: TextIO | None = None,
    db_path: Path = DB_PATH,
    checkpoint_path: Path | str = CHECKPOINT_DB_PATH,
    verifier_today: Callable[[], date] = date.today,
    environ: Mapping[str, str] | None = None,
) -> CliResult:
    """Parse the arguments, run the journey, print what happened, and report how it ended.

    Everything that touches the outside world is an argument (the keyboard, the screen, the
    database and checkpoint files, the clock the Verifier uses, the environment), so a test can run
    the real program in a temporary folder. `main` below fills them in for the real command.
    """
    args = _parser().parse_args(argv)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    scenario_set = load_scenarios()
    scenario = next(s for s in scenario_set.scenarios if s.id == SCENARIOS[args.scenario])
    reference = args.today if args.today is not None else scenario_set.reference_date
    try:
        provider, model_line = _provider(args.fake, scenario, environ)
    except ConfigError as exc:
        err.write(f"Cannot start: {exc}\n")
        return CliResult(1, None)

    if not db_path.exists():
        seed_database(db_path)
        out.write(f"Built the service database from the synthetic dataset: {db_path}\n")
    out.write(_header(scenario, reference, args.scenario, model_line))

    thread = f"cli-{args.scenario}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread}}
    console = Console(input_fn, out)
    tools = default_tools(db_path, today=verifier_today)
    with open_checkpointer(checkpoint_path) as saver:
        graph = build_graph(provider, saver, load_vocabulary(), tools, today=lambda: reference)
        start = GraphState(goal_text=scenario.goal, facts=scenario.facts, contact=DEMO_CONTACT)
        stopped_at: str | None = None
        try:
            result = graph.invoke(start, config)  # type: ignore[arg-type]
            while "__interrupt__" in result:
                answer = console.ask(result["__interrupt__"][0].value)
                result = graph.invoke(Command(resume=answer), config)  # type: ignore[arg-type]
        except NoAnswerError as stop:
            stopped_at = stop.gate
        state = GraphState.model_validate(graph.get_state(config).values)  # type: ignore[arg-type]

    out.write("\n" + render_outcome(state))
    if state.plan:
        out.write("\n" + render_plan(state.plan, state.questions))
    out.write("\n" + render_actions(state.bookings, state.releases, state.withdrawals))
    out.write("\n" + render_trace(state.trace))
    if stopped_at is not None:
        err.write(
            f"Stopped at {stopped_at}: no answer was typed, so nothing was approved. "
            f"The run is saved in {checkpoint_path} (thread {thread}).\n"
        )
        return CliResult(1, state)
    if state.status == "escalated":
        err.write("The request was passed to staff; no plan was approved.\n")
        return CliResult(1, state)
    return CliResult(0, state)


def main() -> int:
    """The `python -m communitybridge.cli` entry point."""
    # A console shows any character, but output piped to a file uses the system code page. Plan
    # statements contain "§", so a stream that cannot encode a character must not crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # a real console or file; a test's stand-in may not
            stream.reconfigure(errors="replace")
    return run().exit_code


if __name__ == "__main__":
    sys.exit(main())
