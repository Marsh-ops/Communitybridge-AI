"""What the three simulated services share: errors, the action log and the fixture reader.

These services stand in for the real transport, accommodation and appointment systems the
prototype must never touch (the core rule: no real bookings, payments or integrations).
Everything they return is labelled SIMULATED, and the "bookings" they make are only requests that
the prototype records in a `SimulatedActionLog`.

Why a booking needs an approval id: the approval gates live in the LangGraph graph,
but the success target is "zero unapproved external actions", and one safeguard is not enough
to promise that. So a simulated service refuses a booking without the id of a recorded approval
event, and the log lets the evaluation count every request and the approval behind it.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from communitybridge.models.consent import KIND_FOR_ACTION
from communitybridge.models.consent_release import DataRelease, ReleaseRecord
from communitybridge.models.simulated import BookingRequest

Model = TypeVar("Model", bound=BaseModel)

_FULL_WEEKDAY = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_SHORT_WEEKDAY = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_PREFIX = {"transport": "TRN", "accommodation": "ACC", "appointment": "APT"}


class SimulatedDataError(Exception):
    """A simulated fixture file is missing, unreadable, or has a row that isn't valid."""


class ApprovalRequiredError(Exception):
    """A simulated booking was requested without the id of a recorded approval event."""


class NotAvailableError(Exception):
    """The route, room or appointment asked for doesn't exist or isn't available then."""


class InvalidRequestError(ValueError):
    """The request itself is malformed, for example zero travellers or a window that ends early."""


def weekday_short(day: date) -> str:
    """'Mon'..'Sun'. Not `strftime`, whose names depend on the computer's language settings."""
    return _SHORT_WEEKDAY[day.weekday()]


def weekday_full(day: date) -> str:
    return _FULL_WEEKDAY[day.weekday()]


def require_approval(approval_id: str) -> str:
    approval = approval_id.strip()
    if not approval:
        raise ApprovalRequiredError("a simulated action needs the id of a recorded approval")
    return approval


class SimulatedActionLog:
    """Every simulated external action the prototype made, in order. Shared by the services.

    Two kinds of action reach the "outside world" in this prototype: a booking request, and (since
    ) a release of the minimum personal data an action needs. Both are refused without the id
    of a recorded approval, both are labelled SIMULATED, and both can be undone when the user
    withdraws consent (`withdraw`).
    """

    def __init__(self) -> None:
        self._requests: list[BookingRequest] = []
        self._releases: list[ReleaseRecord] = []

    @property
    def requests(self) -> tuple[BookingRequest, ...]:
        return tuple(self._requests)

    @property
    def releases(self) -> tuple[ReleaseRecord, ...]:
        return tuple(self._releases)

    def record_release(self, release: DataRelease, approval_id: str) -> ReleaseRecord:
        """Record that the SIMULATED recipient received this data, under a recorded approval."""
        record = ReleaseRecord(
            reference=f"SIM-REL-{len(self._releases) + 1:04d}",
            release=release,
            approval_id=require_approval(approval_id),
        )
        self._releases.append(record)
        return record

    def withdraw(self, action: str) -> tuple[BookingRequest, ...]:
        """Undo one action when consent is withdrawn: remove the details that were
        released for it and cancel the SIMULATED hold it belonged to. Returns what was cancelled.

        A request keeps its reference and its approval id when it is cancelled, so the trace and
        the log still show that it was made and under what. An action with nothing recorded (or
        one that has no booking request, such as sharing with an officer) cancels nothing.
        """
        self._releases = [r for r in self._releases if r.release.action != action]
        kind = KIND_FOR_ACTION.get(action)
        cancelled: list[BookingRequest] = []
        for position, request in enumerate(self._requests):
            if request.kind == kind and request.status == "requested":
                self._requests[position] = request.model_copy(update={"status": "cancelled"})
                cancelled.append(self._requests[position])
        return tuple(cancelled)

    def record(
        self,
        kind: str,
        description: str,
        total_aud: float | None,
        approval_id: str,
    ) -> BookingRequest:
        """Record a request that has already passed every check. Refused requests never get here."""
        number = sum(1 for request in self._requests if request.kind == kind) + 1
        request = BookingRequest(
            kind=kind,  # type: ignore[arg-type]
            reference=f"SIM-{_PREFIX[kind]}-{number:04d}",
            description=f"SIMULATED request: {description}",
            total_aud=total_aud,
            approval_id=approval_id,
        )
        self._requests.append(request)
        return request


def read_fixture_date(path: Path, key: str = "last_verified") -> date:
    """The date a fixture file says it was last checked.

    A plan option cites the simulated routes it uses, and a citation needs a date for the
    Verifier's freshness check, so the file must give one. A missing or malformed date is refused
    by name instead of being guessed.
    """
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimulatedDataError(f"fixture file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SimulatedDataError(f"{path.name} is not valid YAML: {exc}") from exc
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, date) or isinstance(value, datetime):
        raise SimulatedDataError(f"{path.name}: expected a top-level date named '{key}'")
    return value


def read_fixture(path: Path, key: str, model: type[Model]) -> list[Model]:
    """Read a fixture file and validate every row, naming the file and the row if one is bad.

    Each fixture row must itself say `simulated: true` and `synthetic: true` (the record models
    require both), so an unmarked row is refused here and never reaches a response.
    """
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimulatedDataError(f"fixture file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SimulatedDataError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise SimulatedDataError(f"{path.name}: expected a top-level list named '{key}'")
    records = []
    for number, row in enumerate(data[key], start=1):
        try:
            records.append(model.model_validate(row))
        except ValidationError as exc:
            raise SimulatedDataError(f"{path.name}: record {number} is invalid:\n{exc}") from exc
    return records
