"""The Logistics Agent's planning result: the calculator's answer and the options that fit.

`plan_journey` (tools/journey.py) runs the logistics calculator and the
appointment-window matcher for a client the rules engine found eligible, and turns
the windows that work into `LogisticsOption`s the user can choose from. Like every tool result it
is read-only, so no agent can edit a cost after the tool has worked it out.

`rejected` keeps every slot that did not work, with its reasons, because the Coordinator's
re-planning follows those reasons (the design diagram: the next window, then telehealth,). They are
a result too: "no window works" is an answer, not an error.
"""

from pydantic import ConfigDict, Field

from communitybridge.models.common import StrictModel
from communitybridge.models.logistics import LogisticsAssessment
from communitybridge.models.plan import LogisticsOption
from communitybridge.models.windows import RejectedWindow


class JourneyPlan(StrictModel):
    model_config = ConfigDict(frozen=True)

    assessment: LogisticsAssessment
    # Every option is for an in-person appointment; the telehealth fallback is only asked for by
    # the re-planning step. Empty when no slot works, or the travel mode is not decided.
    options: list[LogisticsOption] = Field(default_factory=list)
    rejected: list[RejectedWindow] = Field(default_factory=list)
