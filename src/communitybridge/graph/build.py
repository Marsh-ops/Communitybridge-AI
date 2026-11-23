"""Builds the LangGraph state graph from its nodes. Serves every agent.

The project design makes this the single place where the graph is wired and where each agent's
allowed tools are defined (`AGENT_TOOLS` below). It follows the state graph. So far it holds
the whole path to a booking: extract the goal and Gate 1 (the user confirms the details), Service
Discovery, the Eligibility step, planning the journey, the Verifier, Gate 2 (the user approves the
plan), Gate 3 (the user approves a booking), Gate 4 (the user approves, action by action, the data
each SIMULATED request needs), the SIMULATED booking and then the release of that data. Withdrawing
consent is a node too, applied to a finished run by `withdraw_consent` below. Re-planning adds its nodes here, and the exits that now end the run become edges to them.

The checkpointer is passed in, not created here, so a test can use a temporary file and the web
app can use `data/checkpoints.db`. The tool implementations are passed in too (`default_tools()`
builds the real ones), so a test can hand an agent a fake. Routing decisions are small functions in
the agent modules that read the state and return a plain word; this file maps each word to the
next node.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from communitybridge.agents.consent import (
    CONSENT_ACTIONS,
    CONSENT_DECISION,
    CONSENT_PROMPT,
    CONSENT_REQUEST,
    CONSENT_WITHDRAWAL,
    DATA_RELEASE,
    announce_approve_release,
    make_approve_release,
    make_release_data,
    make_withdraw_consent,
    route_after_release_gate,
)
from communitybridge.agents.coordinator import (
    announce_approve_plan,
    announce_confirm_details,
    approve_plan,
    confirm_details,
    make_extract_goal,
    replan,
    route_after_confirm,
    route_after_extract,
    route_after_plan_gate,
    route_after_replan,
)
from communitybridge.agents.eligibility import (
    RULES_ENGINE,
    make_decide_eligibility,
    route_after_eligibility,
)
from communitybridge.agents.logistics import (
    BOOKING,
    LOGISTICS_PLAN,
    announce_approve_booking,
    approve_booking,
    make_plan_logistics,
    make_request_booking,
    route_after_booking_gate,
    route_after_planning,
    route_after_request,
)
from communitybridge.agents.service_discovery import (
    SERVICE_QUERY,
    make_discover_service,
    route_after_discovery,
)
from communitybridge.agents.verifier import (
    PLAN_VERIFIER,
    make_verify_plan,
    route_after_verification,
)
from communitybridge.config import DB_PATH
from communitybridge.db.seed import load_accommodation
from communitybridge.graph.state import GraphState
from communitybridge.graph.toolbox import ToolBox, ToolFn
from communitybridge.llm.base import LLMProvider
from communitybridge.models import EligibilityExplanation, Service
from communitybridge.models.consent import ConsentRequest
from communitybridge.models.consent_policy import ConsentPolicy
from communitybridge.models.consent_release import (
    ConsentPrompt,
    ConsentSubject,
    ReleaseRecord,
    WithdrawalResult,
)
from communitybridge.models.dataset import Accommodation
from communitybridge.models.discovery import ServiceSearch
from communitybridge.models.engine import EligibilityResult
from communitybridge.models.facts import ConfirmedFacts
from communitybridge.models.journey import JourneyPlan
from communitybridge.models.logistics import LogisticsAssessment
from communitybridge.models.plan import LogisticsOption
from communitybridge.models.replan import DateWindow, Fallback
from communitybridge.models.request import GoalRequest
from communitybridge.models.rules import RuleSet
from communitybridge.models.simulated import BookingRequest
from communitybridge.models.trace import AgentName
from communitybridge.models.verification import PlanCheck
from communitybridge.simulated.services import SimulatedServices, load_simulated_services
from communitybridge.tools.booking import request_journey_booking
from communitybridge.tools.consent import record_decision, request_consent
from communitybridge.tools.consent_flow import (
    actions_for_option,
    prompt_for,
    release_to_sink,
    withdraw_with_sink,
)
from communitybridge.tools.journey import plan_journey
from communitybridge.tools.plan import verify_plan
from communitybridge.tools.policy_loader import load_consent_policy
from communitybridge.tools.rules_engine import EngineContext, decide, load_engine_context
from communitybridge.tools.rules_loader import load_rules
from communitybridge.tools.service_query import ServiceQuery
from communitybridge.tools.vocabulary import Vocabulary

# Who may call what: the one place the permissions are written down. An agent
# is given a `ToolBox` with just these tools, so anything else is refused (`graph/toolbox.py`), and
# the evaluation ("tool-permission violations") checks the trace against this map.
# The Coordinator has no tools: its model and the read-only name lists are constructor arguments
# (a logged deviation,). Each agent is added here when its step is built.
AGENT_TOOLS: dict[AgentName, tuple[str, ...]] = {
    "coordinator": (),
    "service_discovery": (SERVICE_QUERY,),
    "eligibility": (RULES_ENGINE,),
    "logistics": (LOGISTICS_PLAN, BOOKING),
    "verifier": (PLAN_VERIFIER,),
    "consent": (
        CONSENT_ACTIONS,
        CONSENT_PROMPT,
        CONSENT_REQUEST,
        CONSENT_DECISION,
        DATA_RELEASE,
        CONSENT_WITHDRAWAL,
    ),
}


class GraphBuildError(Exception):
    """The graph cannot be built from what it was given."""


def _reached_by_no_edge(state: GraphState) -> dict[str, Any]:
    """The registered `withdraw_consent` node. Withdrawing is applied by `withdraw_consent()`
    below, never by an edge, so reaching this function means the graph was wired wrongly."""
    raise GraphBuildError("withdraw_consent is applied from outside the graph, not by an edge")


def default_tools(
    db_path: Path = DB_PATH,
    rules: RuleSet | None = None,
    simulated: SimulatedServices | None = None,
    today: Callable[[], date] = date.today,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    policy: ConsentPolicy | None = None,
) -> dict[str, ToolFn]:
    """The real tool implementations, keyed by the names used in `AGENT_TOOLS`.

    `simulated` lets a caller (a test, the evaluation) hand in the SIMULATED services so it can read
    their action log afterwards; by default the first tool that needs them loads a fresh set.
    `today` is the date the Verifier judges a source's age against (MAX_SOURCE_AGE_DAYS), so a test
    can fix it and not go stale as the calendar moves. `now` is the time a consent decision or
    withdrawal is stamped with, passed in for the same reason. `policy` is the consent policy
    (`data/policy/data_fields.yaml` by default), loaded when first needed."""
    rule_set = rules if rules is not None else load_rules()

    def service_query(origin_town: str, specialty: str) -> ServiceSearch:
        # A connection per call: it is cheap, read-only, and never shared between threads.
        with ServiceQuery.open(db_path) as query:
            return query.nearest(origin_town, specialty, rule_set)

    @functools.cache
    def engine_context() -> EngineContext:
        # Loaded on the first question, from the same database Service Discovery reads, so the two
        # can never disagree. The rules are the ones passed in, so one setting changes both tools.
        with ServiceQuery.open(db_path) as query:
            return replace(load_engine_context(query), rules=rule_set)

    def rules_engine(
        goal: GoalRequest, facts: ConfirmedFacts, destination_service_id: str
    ) -> EligibilityResult:
        return decide(goal, facts, destination_service_id, engine_context())

    @functools.cache
    def simulated_services() -> SimulatedServices:
        # One set of SIMULATED services, and so one action log, for every tool that uses them.
        return simulated if simulated is not None else load_simulated_services()

    @functools.cache
    def accommodation_records() -> dict[str, Accommodation]:
        return {place.accommodation_id: place for place in load_accommodation()}

    def logistics_plan(
        goal: GoalRequest,
        facts: ConfirmedFacts,
        eligibility: EligibilityResult,
        destination_service_id: str,
        include_fallback: bool = False,
    ) -> JourneyPlan:
        return plan_journey(
            eligibility,
            goal,
            facts,
            destination_service_id,
            simulated_services(),
            engine_context(),
            accommodation_records(),
            include_fallback,
        )

    def simulated_booking(
        option: LogisticsOption,
        service_id: str,
        approval_id: str,
        kinds: list[str] | None = None,
    ) -> list[BookingRequest]:
        return request_journey_booking(option, service_id, approval_id, simulated_services(), kinds)

    consent_policy: list[ConsentPolicy] = [] if policy is None else [policy]

    def the_policy() -> ConsentPolicy:
        # Loaded on first use, so a run that never reaches Gate 4 never needs the file.
        if not consent_policy:
            consent_policy.append(load_consent_policy())
        return consent_policy[0]

    def consent_actions(option: LogisticsOption) -> list[str]:
        return actions_for_option(option)

    def consent_prompt(action: str, subject: ConsentSubject) -> ConsentPrompt:
        return prompt_for(action, subject, the_policy())

    def consent_request(action: str) -> ConsentRequest:
        return request_consent(action, the_policy())

    def consent_decision(request: ConsentRequest, approved: bool) -> ConsentRequest:
        return record_decision(request, approved, now())

    def data_release(
        request: ConsentRequest, subject: ConsentSubject, approval_id: str
    ) -> ReleaseRecord:
        return release_to_sink(
            request, subject, approval_id, the_policy(), simulated_services().log
        )

    def consent_withdrawal(request: ConsentRequest) -> WithdrawalResult:
        return withdraw_with_sink(request, now(), the_policy(), simulated_services().log)

    def plan_verifier(
        service: Service,
        eligibility: EligibilityResult,
        reasons: EligibilityExplanation | None,
        logistics: LogisticsAssessment | None,
        options: list[LogisticsOption],
        fallback: Fallback = "initial",
        window: DateWindow | None = None,
    ) -> PlanCheck:
        return verify_plan(
            service,
            eligibility,
            reasons,
            logistics,
            options,
            engine_context(),
            today(),
            fallback,
            window,
        )

    return {
        SERVICE_QUERY: service_query,
        RULES_ENGINE: rules_engine,
        LOGISTICS_PLAN: logistics_plan,
        BOOKING: simulated_booking,
        PLAN_VERIFIER: plan_verifier,
        CONSENT_ACTIONS: consent_actions,
        CONSENT_PROMPT: consent_prompt,
        CONSENT_REQUEST: consent_request,
        CONSENT_DECISION: consent_decision,
        DATA_RELEASE: data_release,
        CONSENT_WITHDRAWAL: consent_withdrawal,
    }


def toolbox_for(agent: AgentName, tools: Mapping[str, ToolFn]) -> ToolBox:
    """The box for one agent: only the tools `AGENT_TOOLS` lists for it, taken from `tools`."""
    given: dict[str, ToolFn] = {}
    for name in AGENT_TOOLS[agent]:
        if name not in tools:
            raise GraphBuildError(f"the {agent} agent needs the tool {name!r}, which was not given")
        given[name] = tools[name]
    return ToolBox(agent, given)


def build_graph(
    provider: LLMProvider,
    checkpointer: BaseCheckpointSaver,
    vocabulary: Vocabulary,
    tools: Mapping[str, ToolFn],
    *,
    today: Callable[[], date] = date.today,
) -> CompiledStateGraph:
    """Compile the graph. Start a run with `graph.invoke(GraphState(goal_text=...), config)` where
    `config` holds a `thread_id`; each journey uses its own thread.

    `vocabulary` (`tools.vocabulary.load_vocabulary()` in the app) is the read-only list of
    specialty and town names the Coordinator is shown. `tools` (`default_tools()` in the app) holds
    the implementation of each tool; each agent is handed only the ones `AGENT_TOOLS` lists for it.
    Both are passed in here, in the one place that wires the agents, and not looked up by the
    agents while they run."""
    graph = StateGraph(GraphState)
    graph.add_node("extract_goal", make_extract_goal(provider, vocabulary, today))
    graph.add_node("announce_confirm_details", announce_confirm_details)
    graph.add_node("confirm_details", confirm_details)
    graph.add_node(
        "discover_service", make_discover_service(toolbox_for("service_discovery", tools))
    )
    graph.add_node(
        "decide_eligibility",
        make_decide_eligibility(provider, toolbox_for("eligibility", tools)),
    )
    # One tool box for the Logistics Agent, shared by its planning and booking nodes.
    logistics_tools = toolbox_for("logistics", tools)
    graph.add_node("plan_logistics", make_plan_logistics(logistics_tools))
    graph.add_node("announce_approve_booking", announce_approve_booking)
    graph.add_node("approve_booking", approve_booking)
    graph.add_node("request_booking", make_request_booking(logistics_tools))
    graph.add_node("verify_plan", make_verify_plan(toolbox_for("verifier", tools)))
    graph.add_node("announce_approve_plan", announce_approve_plan)
    graph.add_node("approve_plan", approve_plan)
    graph.add_node("replan", replan)
    # One tool box for the Consent Agent, shared by Gate 4 and the release.
    consent_tools = toolbox_for("consent", tools)
    graph.add_node("announce_approve_release", announce_approve_release)
    graph.add_node("approve_release", make_approve_release(consent_tools))
    graph.add_node("release_data", make_release_data(consent_tools))

    graph.add_edge(START, "extract_goal")
    graph.add_conditional_edges(
        "extract_goal",
        route_after_extract,
        {"announce": "announce_confirm_details", "finish": END},
    )
    graph.add_edge("announce_confirm_details", "confirm_details")
    graph.add_conditional_edges(
        "confirm_details",
        route_after_confirm,
        {"extract": "extract_goal", "discover": "discover_service", "finish": END},
    )
    graph.add_conditional_edges(
        "discover_service",
        route_after_discovery,
        {"decide": "decide_eligibility", "finish": END},
    )
    graph.add_conditional_edges(
        "decide_eligibility",
        route_after_eligibility,
        # An eligible client is planned for; one who is not eligible, or whose case needs review,
        # gets an information-only plan, which is checked and approved like any other.
        {"plan": "plan_logistics", "inform": "verify_plan", "finish": END},
    )
    graph.add_conditional_edges(
        "plan_logistics",
        route_after_planning,
        # No arrangement works: the Coordinator re-plans.
        {"verify": "verify_plan", "replan": "replan", "finish": END},
    )
    graph.add_conditional_edges(
        "verify_plan",
        route_after_verification,
        # A blocked statement the plan rests on: the Coordinator re-plans.
        {"approve": "announce_approve_plan", "replan": "replan", "finish": END},
    )
    graph.add_edge("announce_approve_plan", "approve_plan")
    graph.add_conditional_edges(
        "approve_plan",
        route_after_plan_gate,
        # The user rejected the plan: that counts as a failed attempt, so the Coordinator re-plans.
        {"book": "announce_approve_booking", "done": END, "replan": "replan"},
    )
    # A re-plan goes back to planning the journey, with the next fallback; after the fallbacks are
    # used (or when telehealth may not be offered) the run goes to staff. The service and the
    # decision are not asked for again: re-planning never changes the PATS decision.
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {"plan": "plan_logistics", "finish": END},
    )
    graph.add_edge("announce_approve_booking", "approve_booking")
    graph.add_conditional_edges(
        "approve_booking",
        route_after_booking_gate,
        # Gate 3 no longer leads straight to the booking: Gate 4 decides what may be sent.
        {"consent": "announce_approve_release", "finish": END},
    )
    graph.add_edge("announce_approve_release", "approve_release")
    graph.add_conditional_edges(
        "approve_release",
        route_after_release_gate,
        {"book": "request_booking", "finish": END},
    )
    # The booking is all-or-nothing and can fail, so it comes first: no details leave for a request
    # that could not be made.
    graph.add_conditional_edges(
        "request_booking",
        route_after_request,
        {"release": "release_data", "finish": END},
    )
    graph.add_edge("release_data", END)
    # Withdrawing consent is the user's choice, at any time after approving, so no edge leads here.
    # The node is registered so that `withdraw_consent` below can record its update "as" it; the
    # update itself is made by the node that function builds for the action being withdrawn.
    graph.add_node("withdraw_consent", _reached_by_no_edge)
    graph.add_edge("withdraw_consent", END)
    return graph.compile(checkpointer=checkpointer)


def withdraw_consent(
    graph: CompiledStateGraph, config: Mapping[str, Any], tools: Mapping[str, ToolFn], action: str
) -> GraphState:
    """Withdraw consent for one action on a finished run, and return the new state.

    The user may withdraw at any time after approving, so this is applied to a run that has ended,
    not reached by an edge. The Consent Agent's withdrawal node works out the update (the consent
    becomes withdrawn, the released details are removed, the SIMULATED hold is cancelled, the user
    is told what that undoes, and the trace gets a `withdrawn` event) and it is saved as that node's
    own update. A run that is still waiting at a gate is refused: its answer comes first.
    """
    snapshot = graph.get_state(config)  # type: ignore[arg-type]
    if snapshot.next:
        raise GraphBuildError("a run that is still waiting at a gate cannot have consent withdrawn")
    state = GraphState.model_validate(snapshot.values)
    node = make_withdraw_consent(toolbox_for("consent", tools), action)
    graph.update_state(config, node(state), as_node="withdraw_consent")  # type: ignore[arg-type]
    return GraphState.model_validate(graph.get_state(config).values)  # type: ignore[arg-type]
