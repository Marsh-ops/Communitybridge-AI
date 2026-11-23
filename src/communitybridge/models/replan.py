"""Schema for the re-planning scenarios (data/scenarios/replan_scenarios.yaml).

The eligibility scenarios (models/scenario.py) mark what the rules engine decides. These mark
what the Coordinator does when a plan can't be put together: the design allows only fixed
fallbacks, "the next appointment window, then telehealth", and escalation "to staff after two
failed re-plans". The validators make a scenario that breaks those rules impossible to
write, so 's failure-path tests can trust the expected path.

 adds what the running graph needs to follow the same rules: the dates each attempt looks at
(`DateWindow`, `window_for`) and the record of an attempt that failed (`FailedAttempt`). A "window"
is a range of dates, because the planning tool looks at every appointment slot inside the dates it
is given. The *next* window is the client's own range moved forward by its own length.
"""

from datetime import date, datetime, timedelta
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel
from communitybridge.models.request import GoalRequest
from communitybridge.models.scenario import Outcome, ScenarioFacts

Fallback = Literal["initial", "next_window", "telehealth"]
# The one permitted order. "initial" is the first plan; the other two are re-plans.
FALLBACK_ORDER: tuple[Fallback, ...] = ("initial", "next_window", "telehealth")
MAX_REPLANS = len(FALLBACK_ORDER) - 1  # 2, then escalate


class DateWindow(StrictModel):
    """A range of dates the planner looks at, both ends included."""

    model_config = ConfigDict(frozen=True)

    earliest_date: date
    latest_date: date

    @model_validator(mode="after")
    def _dates_in_order(self) -> Self:
        if self.latest_date < self.earliest_date:
            raise ValueError("latest_date must be on or after earliest_date")
        return self

    @property
    def days(self) -> int:
        return (self.latest_date - self.earliest_date).days + 1


def window_after(window: DateWindow) -> DateWindow:
    """The next window: starts the day after this one ends, and is as long."""
    start = window.latest_date + timedelta(days=1)
    return DateWindow(earliest_date=start, latest_date=start + timedelta(days=window.days - 1))


def client_window(goal: GoalRequest) -> DateWindow:
    """The dates the client confirmed at Gate 1."""
    return DateWindow(earliest_date=goal.earliest_date, latest_date=goal.latest_date)


def window_for(goal: GoalRequest, fallback: Fallback) -> DateWindow:
    """The dates an attempt looks at. The first plan and the telehealth attempt use the client's own
    dates (the specialist's telehealth slot is offered inside the window they asked for,);
    the next-window attempt looks at the dates after them."""
    own = client_window(goal)
    return window_after(own) if fallback == "next_window" else own


# Why an attempt was given up on. "no_option": planning found no arrangement that works.
# "blocked": the Verifier blocked a statement the plan rests on. "rejected": the user rejected the
# plan at Gate 2. Each counts as a failed attempt, so a user can never cause an endless loop.
WhyFailed = Literal["no_option", "blocked", "rejected"]


class FailedAttempt(StrictModel):
    """An attempt that did not lead to a plan the user approved, as the state keeps it."""

    model_config = ConfigDict(frozen=True)

    fallback: Fallback
    window: DateWindow
    why: WhyFailed


class PlanAttempt(StrictModel):
    """One attempt to build a plan, as the trace will show it."""

    fallback: Fallback
    service_id: NonEmptyStr
    mode: Literal["in_person", "telehealth"]
    slot: datetime
    result: Literal["failed", "planned"]
    tool_or_source: NonEmptyStr  # the tool whose result decided this attempt
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _mode_matches_fallback(self) -> Self:
        if (self.fallback == "telehealth") != (self.mode == "telehealth"):
            raise ValueError("only the telehealth fallback uses telehealth mode")
        return self

    @property
    def rejected_by_user(self) -> bool:
        """True if the client turned this plan down at Gate 2 (its `tool_or_source` is the user)."""
        return self.tool_or_source.startswith("user")


class ReplanScenario(StrictModel):
    id: str = Field(pattern=r"^RP-\d{2}$")
    title: NonEmptyStr
    goal: NonEmptyStr
    expected_fields: GoalRequest
    facts: ScenarioFacts = Field(default_factory=ScenarioFacts)
    destination_service_id: NonEmptyStr
    # Re-planning never changes the PATS decision; it only changes how the plan is delivered.
    expected_eligibility: Outcome
    attempts: list[PlanAttempt] = Field(min_length=1, max_length=len(FALLBACK_ORDER))
    # Set only on a path that goes to staff after ONE re-plan because telehealth may not be offered
    # "plan_rejected" (a plan the client turned down proves an in-person
    # option exists) or "client_can_drive" (a private vehicle, §7.2.1, no timetable can rule out).
    telehealth_not_offered: Literal["plan_rejected", "client_can_drive"] | None = None
    expected_status: Literal["planned", "escalated"]
    expected_replan_count: int = Field(ge=0, le=MAX_REPLANS)
    must_cite: list[str] = Field(min_length=1)
    working: NonEmptyStr
    synthetic: Literal[True]

    @model_validator(mode="after")
    def _follows_the_fallback_rules(self) -> Self:
        fallbacks = tuple(a.fallback for a in self.attempts)
        if fallbacks != FALLBACK_ORDER[: len(fallbacks)]:
            raise ValueError(f"fallbacks must run in the order {FALLBACK_ORDER}, got {fallbacks}")
        if any(a.result == "planned" for a in self.attempts[:-1]):
            raise ValueError("no attempt may follow after a plan has succeeded")
        if self.expected_replan_count != len(self.attempts) - 1:
            raise ValueError("expected_replan_count must equal the number of re-plans attempted")
        last = self.attempts[-1]
        if last.result == "failed" and len(self.attempts) < len(FALLBACK_ORDER):
            if not self._may_end_early():
                raise ValueError("a failed attempt must be followed by the next fallback")
        elif self.telehealth_not_offered is not None:
            raise ValueError("telehealth_not_offered is only for a path that ends early")
        expected = "planned" if last.result == "planned" else "escalated"
        if self.expected_status != expected:
            raise ValueError(f"expected_status must be {expected!r} for this path")
        return self

    def _may_end_early(self) -> bool:
        """A path may stop after the next window only for a reason telehealth was not offered.

        The reason has to be true of the scenario, so the file cannot excuse an early end it does
        not earn: a rejected plan must really be in the attempts, a vehicle must really be there."""
        if self.telehealth_not_offered is None or self.attempts[-1].fallback != "next_window":
            return False
        if self.telehealth_not_offered == "plan_rejected":
            if not any(a.rejected_by_user for a in self.attempts):
                raise ValueError("plan_rejected needs an attempt the user rejected at Gate 2")
            return True
        if not self.facts.has_private_vehicle:
            raise ValueError("client_can_drive needs a client with a private vehicle (§7.2.1)")
        return True


class ReplanScenarioSet(StrictModel):
    reference_date: date
    rules_version: NonEmptyStr
    decision_records: list[str] = Field(default_factory=list)
    scenarios: list[ReplanScenario] = Field(min_length=1)
