"""The SQLite checkpointer: saving the graph state so a run can pause and resume.

The design makes every approval gate a LangGraph `interrupt` and says "a SQLite checkpointer
saves the state at each gate so a run can pause for the user and resume later"
(design notes). The user may take minutes or days to answer, and the
web app may restart in between, so the state has to be in a file and not in memory.

Three decisions here:
- **The file is separate from the dataset.** `service_query` opens `DB_PATH` read-only; the graph
  writes to `CHECKPOINT_DB_PATH`, so nothing it saves can reach the curated data.
- **The serializer has an allow-list.** LangGraph rebuilds Python classes from the checkpoint file,
  and its documentation warns that anyone who can write to that file may be able to run code if
  any class is allowed. Only this project's own state models are allowed here. By default
  LangGraph 1.2.x allows everything and warns that this "will be blocked in a future version".
- **Opening fails fast and closes cleanly.** A bad path or a file that isn't a database raises
  `CheckpointError` at once, not at the first save, and the connection is always closed (an open
  handle stops the file being moved or deleted on Windows).
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import get_args

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from communitybridge.config import CHECKPOINT_DB_PATH
from communitybridge.graph.state import GraphState


class CheckpointError(Exception):
    """The checkpoint database could not be opened."""


def state_model_classes() -> set[type[BaseModel]]:
    """Every model class that can appear inside a GraphState, found by walking its fields.

    Walking the fields (instead of listing the classes by hand) means a model added to the state
    later is allowed automatically, and a test fails until that is checked on purpose.
    """
    found: set[type[BaseModel]] = set()

    def walk(annotation: object) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in found:
                return
            found.add(annotation)
            for field in annotation.model_fields.values():
                walk(field.annotation)
        # Looks inside list[X], X | None and similar. Plain types such as date have no arguments.
        for argument in get_args(annotation):
            walk(argument)

    walk(GraphState)
    return found


def graph_serde() -> JsonPlusSerializer:
    """The checkpoint serializer, allowed to rebuild only the state models.

    A class outside the list comes back as plain data (a dict), never as an instance, and
    LangGraph logs that it was blocked. Pickle is off, which is LangGraph's default.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=list(state_model_classes()))


@contextmanager
def open_checkpointer(path: Path | str = CHECKPOINT_DB_PATH) -> Iterator[SqliteSaver]:
    """Open the checkpoint file for the length of a `with` block, then close it.

    Use a new block (a new connection) to resume a run later; the file holds everything.
    `check_same_thread=False` is needed because LangGraph may run a node on another thread;
    SqliteSaver guards the connection with its own lock.
    """
    try:
        connection = sqlite3.connect(str(path), check_same_thread=False)
    except sqlite3.Error as exc:
        raise CheckpointError(f"cannot open the checkpoint database at {path}: {exc}") from exc
    try:
        saver = SqliteSaver(connection, serde=graph_serde())
        # SqliteSaver would otherwise create its tables on first use. Doing it now means a file
        # that is not a database fails here, with a clear message, and not in the middle of a run.
        saver.setup()
    except sqlite3.Error as exc:
        connection.close()
        raise CheckpointError(f"cannot use {path} as a checkpoint database: {exc}") from exc
    try:
        yield saver
    finally:
        connection.close()
