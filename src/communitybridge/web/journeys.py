"""Runs journeys through the agent graph for the web pages.

Serves all six agents, as the command line does (`cli.py`). It builds the same graph from
`graph/build.py` with the real tools, rules and SIMULATED services, and keeps each journey as one
thread in the SQLite checkpoint file. That file is what lets a journey pause at an approval gate
while the browser is closed or the server restarts, and carry on when the person answers
(project plan: approval gates use LangGraph `interrupt` with a SQLite checkpointer,).

Why a service object and not code in the routes:
- The routes stay thin and only translate HTTP into calls here, so the rule "a gate is answered by
  a POSTed form the server records" lives in one place and tests can check it without a browser.
- Everything that touches the outside world (the model, the database files, the clock) is an
  argument, so a test builds a service in a temporary folder with the FakeProvider.

Each request builds the graph, uses it and closes the checkpoint connection, like one CLI run.
That is simple and safe across the threads FastAPI runs routes on. A journey is not locked, so two
browser tabs answering the same gate at the same moment is not handled (a prototype limit).
"""

import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.types import Command

from communitybridge.agents.consent import ConsentAgentError
from communitybridge.config import (
    CHECKPOINT_DB_PATH,
    DB_PATH,
    ConfigError,
    ollama_settings,
    reference_date_override,
)
from communitybridge.db.seed import seed_database
from communitybridge.graph.build import (
    GraphBuildError,
    build_graph,
    default_tools,
    withdraw_consent,
)
from communitybridge.graph.checkpoint import open_checkpointer
from communitybridge.graph.state import GraphState
from communitybridge.llm.base import LLMProvider
from communitybridge.llm.ollama import OllamaProvider
from communitybridge.simulated.services import SimulatedServices, load_simulated_services
from communitybridge.tools.consent import ConsentError
from communitybridge.tools.vocabulary import load_vocabulary

# A journey id is 32 lower-case hex characters from `uuid4`, unguessable enough for a prototype
# that holds only synthetic data. Anything else is refused before it reaches the database.
_JOURNEY_ID = re.compile(r"[0-9a-f]{32}")


class JourneyNotFoundError(Exception):
    """There is no journey with that id."""


class SetupProblemError(Exception):
    """The app is not set up to run a journey (for example no model name), with a message."""


class WithdrawalRefusedError(Exception):
    """Consent cannot be withdrawn now: not approved, already withdrawn, or the run is waiting."""


@dataclass(frozen=True)
class JourneyView:
    """Where a journey is: its state, and the question it is waiting on (None if none)."""

    state: GraphState
    gate: Mapping[str, Any] | None


class JourneyService:
    """Starts, reads and answers journeys. See the module docstring."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        environ: Mapping[str, str] | None = None,
        db_path: Path = DB_PATH,
        checkpoint_path: Path | str = CHECKPOINT_DB_PATH,
        today: Callable[[], date] | None = None,
        verifier_today: Callable[[], date] = date.today,
    ) -> None:
        self._provider = provider
        self._environ = environ
        self._db_path = db_path
        self._checkpoint_path = checkpoint_path
        self._today = today
        self._verifier_today = verifier_today
        # One set of SIMULATED services per journey, kept for as long as the server runs. A booking
        # request made at Gate 4 is held in that set's log, and withdrawing consent later (a
        # different request) has to find and cancel it there. A fresh set per request would make a
        # withdrawal cancel nothing. Per journey, not shared, so one journey's requests never use
        # up another's SIMULATED seats. Limit: after a server restart a journey gets new services,
        # so a withdrawal removes the released details and the consent but cannot cancel the hold.
        self._simulated: dict[str, SimulatedServices] = {}
        self._simulated_lock = threading.Lock()

    # --- settings ------------------------------------------------------------------------

    def reference_today(self) -> date:
        """The date "next month" is worked out from: the setting, else the real date."""
        if self._today is not None:
            return self._today()
        try:
            return reference_date_override(self._environ) or date.today()
        except ConfigError as exc:
            raise SetupProblemError(str(exc)) from exc

    def _model(self) -> LLMProvider:
        if self._provider is None:
            try:
                self._provider = OllamaProvider(ollama_settings(self._environ))
            except ConfigError as exc:
                raise SetupProblemError(str(exc)) from exc
        return self._provider

    def _ensure_database(self) -> None:
        # Built on first use, as the CLI does, so creating the app never writes a file.
        if not self._db_path.exists():
            seed_database(self._db_path)

    def _services_for(self, journey_id: str) -> SimulatedServices:
        with self._simulated_lock:
            if journey_id not in self._simulated:
                self._simulated[journey_id] = load_simulated_services()
            return self._simulated[journey_id]

    def _tools(self, journey_id: str) -> Any:
        return default_tools(
            self._db_path,
            simulated=self._services_for(journey_id),
            today=self._verifier_today,
        )

    @contextmanager
    def _graph(self, journey_id: str) -> Iterator[Any]:
        model = self._model()
        reference = self.reference_today()
        self._ensure_database()
        with open_checkpointer(self._checkpoint_path) as saver:
            yield build_graph(
                model,
                saver,
                load_vocabulary(),
                self._tools(journey_id),
                today=lambda: reference,
            )

    # --- journeys ------------------------------------------------------------------------

    @staticmethod
    def _config(journey_id: str) -> dict[str, Any]:
        if not _JOURNEY_ID.fullmatch(journey_id):
            raise JourneyNotFoundError(journey_id)
        return {"configurable": {"thread_id": f"web-{journey_id}"}}

    @staticmethod
    def _read(graph: Any, config: dict[str, Any]) -> JourneyView:
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise JourneyNotFoundError(config["configurable"]["thread_id"])
        gate = snapshot.interrupts[0].value if snapshot.interrupts else None
        return JourneyView(GraphState.model_validate(snapshot.values), gate)

    def start(self, goal_text: str) -> tuple[str, JourneyView]:
        """Begin a journey from the person's words and run it to its first question."""
        journey_id = uuid.uuid4().hex
        config = self._config(journey_id)
        with self._graph(journey_id) as graph:
            graph.invoke(GraphState(goal_text=goal_text), config)
            return journey_id, self._read(graph, config)

    def view(self, journey_id: str) -> JourneyView:
        config = self._config(journey_id)
        with self._graph(journey_id) as graph:
            return self._read(graph, config)

    def withdraw(self, journey_id: str, action: str) -> JourneyView:
        """Withdraw consent for one approved action on a finished journey.

        The Consent Agent's own withdrawal does the work: the consent becomes withdrawn, the details
        released for that action are removed and its SIMULATED request is cancelled.
        """
        config = self._config(journey_id)
        with self._graph(journey_id) as graph:
            self._read(graph, config)  # 404 for an unknown journey
            try:
                withdraw_consent(graph, config, self._tools(journey_id), action)
            except (GraphBuildError, ConsentError, ConsentAgentError) as exc:
                raise WithdrawalRefusedError(str(exc)) from exc
            return self._read(graph, config)

    def answer(self, journey_id: str, answer: Mapping[str, Any]) -> JourneyView:
        """Hand the person's answer to the gate that is waiting and run to the next stop."""
        config = self._config(journey_id)
        with self._graph(journey_id) as graph:
            self._read(graph, config)  # 404 for an unknown journey, before anything is resumed
            graph.invoke(Command(resume=dict(answer)), config)
            return self._read(graph, config)
