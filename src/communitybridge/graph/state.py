"""GraphState: the one object that moves through the LangGraph graph. Serves every agent.

The fields are the ones in the data model: the goal and the confirmed facts, the service
found, the rules engine's complete result and its plain-language explanation, the logistics
result and options, the user's choice and the simulated bookings, the plan items, the consent
records, the re-plan count, the run status and the trace.
It is a strict Pydantic model like every other boundary model, so an agent that invents a field
(such as `eligible`) is refused.

These rules are enforced on the data itself, not just in a prompt:
- Nothing downstream of Gate 1 can exist until the user has confirmed the extracted goal.
- A decision is about a service, and the reasons and the model's explanation are about a decision,
  so none of them can exist without the thing before it.
- Nothing is planned, chosen or booked for a client the rules engine did not find eligible, a choice must be one of the options, and a booking needs a choice.
- A plan reports a decision, so it cannot exist without the eligibility result it reports; and a plan item is "verified" only with a source, by `PlanItem` itself.
- Data is released and a booking is requested only for an action the user approved (Gate 4,
  ): a release or a booking cannot exist without an approved consent for its action, and a
  withdrawal notice cannot exist without the withdrawn consent it is about.
- At most 2 re-plans, the same limit as `models/replan.py` (the design diagram: "to staff after two
  failed re-plans"), and only along the fallbacks in the one order the design allows: the first
  plan, then the next window, then telehealth. The attempt under way follows from how many
  re-plans have been made, and the failed attempts so far are exactly the ones before it.

`trace` carries a reducer, so each node returns only its new events and LangGraph appends them.
Every other field is replaced by what a node returns.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from communitybridge.models import (
    ConsentRequest,
    EligibilityExplanation,
    GoalRequest,
    LogisticsOption,
    PlanItem,
    Service,
    StrictModel,
    TraceEvent,
)
from communitybridge.models.common import NonEmptyStr
from communitybridge.models.consent import ACTION_FOR_KIND
from communitybridge.models.consent_release import ReleaseRecord, WithdrawalNotice
from communitybridge.models.contact import ClientContact
from communitybridge.models.engine import EligibilityResult
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.logistics import LogisticsAssessment
from communitybridge.models.replan import FALLBACK_ORDER, MAX_REPLANS, FailedAttempt, Fallback
from communitybridge.models.simulated import BookingRequest
from communitybridge.trace.recorder import append_trace

# "planned" and "escalated" are the two end states the re-planning scenarios already use
# (`expected_status` in data/scenarios/replan_scenarios.yaml). "declined" is a user saying no at a
# gate, which takes no action. "awaiting_approval" means a gate is waiting for the user.
RunStatus = Literal["in_progress", "awaiting_approval", "planned", "declined", "escalated"]


class GraphState(StrictModel):
    # The user's own words, kept so the Coordinator can read them and so Gate 1 can offer
    # "edit". It is personal data, so it never goes into a trace event or an error message.
    goal_text: NonEmptyStr | None = None
    goal: GoalRequest | None = None
    # Details the client or coordinator confirms beyond the goal text (models/facts.py). They are
    # input, given when the run starts, and the defaults describe a typical adult client. Gate 1
    # shows them to the user, so an assumption is never hidden.
    facts: ConfirmedFacts = Field(default_factory=ConfirmedFacts)
    # The client's own contact details. Input, like `facts`; synthetic only.
    # Only the Consent step reads it, and it is shown to the user at Gate 4 before any of it is
    # released. It is personal data, so it never goes into a trace event or an error message.
    contact: ClientContact | None = None
    service: Service | None = None
    # The rules engine's complete answer: the `Decision` (read-only, with `eligible`, the rule hits
    # and the rules version) plus the outcome, notices and entitlements that Logistics and the
    # plan need. Only the rules engine writes it.
    eligibility: EligibilityResult | None = None
    # The rules engine's own reasons for that result, in plain words, built by the rules and never
    # by a model (`fallback_explanation`). This is what the plan states as verified.
    reasons: EligibilityExplanation | None = None
    # The wording a language model gave for that result, if it wrote one that passed the
    # Eligibility Agent's check; None when the model was down or kept failing it. It is shown to
    # the user apart from the plan, labelled, and is never a plan statement: the check covers the
    # sections it cites and the outcome it states, not what it says about the client.
    explanation: EligibilityExplanation | None = None
    # The logistics calculator's whole answer (travel mode, costs, entitlements, questions) and the
    # options that fit the client's dates. Only an eligible client gets either.
    logistics: LogisticsAssessment | None = None
    options: list[LogisticsOption] = Field(default_factory=list)
    # The option the user picked at Gate 3, as a position in `options`. Only the user sets it.
    chosen_option: int | None = Field(default=None, ge=0)
    # The SIMULATED booking requests made for that option, under the Gate 3 approval.
    bookings: list[BookingRequest] = Field(default_factory=list)
    # The plan the user is shown: every statement the Verifier checked, "verified" (it has sources
    # that are current) or "blocked" (with the reason). Written only by the Verifier.
    plan: list[PlanItem] = Field(default_factory=list)
    # What the user is asked to answer or confirm at the plan gate: stale or conflicting
    # information the Verifier found, and the questions the tools raised (an entitlement that
    # depends on an answer, a travel mode PATS does not name).
    questions: list[NonEmptyStr] = Field(default_factory=list)
    # True if a statement the plan rests on was blocked, so the plan goes back to be re-planned.
    replan_needed: bool = False
    # The user's answer at the plan gate (Gate 2): True approves the plan, False rejects it.
    plan_approved: bool | None = None
    # One consent request per action the chosen option needs, decided at Gate 4. A decision
    # is final; a withdrawal replaces the record with a withdrawn copy.
    consents: list[ConsentRequest] = Field(default_factory=list)
    # The data released to the SIMULATED recipients, under the Gate 4 approvals. Withdrawing
    # an action removes its release.
    releases: list[ReleaseRecord] = Field(default_factory=list)
    # What the user was told when they withdrew consent for an action.
    withdrawals: list[WithdrawalNotice] = Field(default_factory=list)
    replan_count: int = Field(default=0, ge=0, le=MAX_REPLANS)
    # The attempt under way: "initial", then "next_window" after the first re-plan, then
    # "telehealth" after the second. Only the Coordinator's `replan` step changes it.
    fallback: Fallback = "initial"
    # The attempts that failed so far, in order. After a re-plan there is one per re-plan made;
    # an escalated run has one more, the attempt the Coordinator gave up after.
    attempts: list[FailedAttempt] = Field(default_factory=list)
    status: RunStatus = "in_progress"
    trace: Annotated[list[TraceEvent], append_trace] = Field(default_factory=list)

    @model_validator(mode="after")
    def _results_need_a_confirmed_goal(self) -> Self:
        has_results = bool(
            self.service is not None
            or self.eligibility is not None
            or self.reasons is not None
            or self.explanation is not None
            or self.logistics is not None
            or self.options
            or self.chosen_option is not None
            or self.bookings
            or self.plan
            or self.questions
            or self.replan_needed
            or self.plan_approved is not None
            or self.releases
            or self.withdrawals
        )
        if has_results and not (self.goal is not None and self.goal.confirmed_by_user):
            raise ValueError(
                "service, eligibility, reasons, explanation, logistics, options, chosen_option, "
                "bookings "
                "and plan need a goal the user has confirmed (Gate 1)"
            )
        return self

    @model_validator(mode="after")
    def _each_result_needs_the_one_before(self) -> Self:
        if self.eligibility is not None and self.service is None:
            raise ValueError("an eligibility result needs the service it is about")
        if self.reasons is not None and self.eligibility is None:
            raise ValueError("the reasons need the eligibility result they come from")
        if self.explanation is not None and self.eligibility is None:
            raise ValueError("an explanation needs the eligibility result it explains")
        if (self.plan or self.questions) and self.eligibility is None:
            raise ValueError("a plan needs the eligibility result it reports")
        if (self.replan_needed or self.plan_approved is not None) and not self.plan:
            raise ValueError("a verdict on the plan and the user's answer need the plan itself")
        return self

    @model_validator(mode="after")
    def _data_and_bookings_only_for_approved_actions(self) -> Self:
        # Gate 4 in the data: whatever an agent tries to write, a release or a booking
        # request cannot be in the state without the user's approval of its action. An approval
        # that was later withdrawn still counts: the record shows it was given.
        approved = {c.action for c in self.consents if c.approved is True}
        for record in self.releases:
            if record.release.action not in approved:
                raise ValueError("a release needs an approved consent for its action")
        for booking in self.bookings:
            if ACTION_FOR_KIND[booking.kind] not in approved:
                raise ValueError("a booking request needs an approved consent for its action")
        withdrawn = {c.action for c in self.consents if c.status == "withdrawn"}
        for notice in self.withdrawals:
            if notice.action not in withdrawn:
                raise ValueError("a withdrawal notice needs the withdrawn consent it is about")
        return self

    @model_validator(mode="after")
    def _replans_follow_the_fallback_order(self) -> Self:
        # Design: "the next appointment window, then telehealth", after two failed re-plans to staff.
        if self.fallback != FALLBACK_ORDER[self.replan_count]:
            raise ValueError("the attempt under way must follow from the number of re-plans made")
        tried = tuple(a.fallback for a in self.attempts)
        if tried != FALLBACK_ORDER[: len(tried)]:
            raise ValueError("failed attempts must run in the order the design allows")
        if len(self.attempts) not in (self.replan_count, self.replan_count + 1):
            raise ValueError("there is one failed attempt per re-plan, or one more once escalated")
        if len(self.attempts) == self.replan_count + 1 and self.status != "escalated":
            raise ValueError("a run that gave up on its last attempt must be escalated")
        return self

    @model_validator(mode="after")
    def _logistics_only_for_an_eligible_client(self) -> Self:
        # The core rule in the data: nothing is planned, chosen or booked for a client the rules
        # engine did not find eligible, whatever an agent tries to write.
        planned = bool(
            self.logistics is not None or self.options or self.chosen_option is not None
        ) or bool(self.bookings)
        eligible = self.eligibility is not None and self.eligibility.outcome == "eligible"
        if planned and not eligible:
            raise ValueError("logistics, options and bookings need an eligible client")
        if self.options and self.logistics is None:
            raise ValueError("options need the logistics result they come from")
        return self

    @model_validator(mode="after")
    def _a_choice_and_its_bookings(self) -> Self:
        if self.chosen_option is not None and self.chosen_option >= len(self.options):
            raise ValueError("chosen_option must be a position in options")
        if self.bookings and self.chosen_option is None:
            raise ValueError("bookings are made for a chosen option")
        return self
