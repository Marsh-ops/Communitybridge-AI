"""Consent tools for the graph: which actions a journey needs, what each shares, releasing, and
withdrawing.

`tools/consent.py` holds the rules for one consent request. This module joins them to the
journey the user chose and to the SIMULATED action log, so the Consent Agent can do its whole job
through its tool box without ever reading a personal value itself:

- `actions_for_option` says which actions the chosen option needs (the same kinds the booking step
  will make, from the one function that decides them);
- `available_values` fills in the policy's fields from the goal, the service, the option and the
  client's contact details. It offers only fields that are in the policy's catalogue, and nothing
  the policy marks sensitive or never-share, whatever else the state holds;
- `prompt_for` is the consent screen, and `release_to_sink` is the release: both use the same
  values, so what the user approved is what is sent;
- `withdraw_with_sink` withdraws consent and undoes the action in the SIMULATED log: the released
  details are removed and the SIMULATED hold is cancelled.

Nothing is really sent anywhere. Errors name fields, never values.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta

from communitybridge.models.consent import ACTION_FOR_KIND, ConsentRequest
from communitybridge.models.consent_policy import ConsentPolicy
from communitybridge.models.consent_release import (
    ConsentPrompt,
    ConsentSubject,
    ReleaseRecord,
    WithdrawalResult,
)
from communitybridge.models.plan import LogisticsOption
from communitybridge.simulated.common import SimulatedActionLog
from communitybridge.tools.booking import kinds_for_option
from communitybridge.tools.consent import describe_action, release_data, withdraw_consent


def actions_for_option(option: LogisticsOption) -> list[str]:
    """The consent actions the option needs, in the order the requests are made."""
    return [ACTION_FOR_KIND[kind] for kind in kinds_for_option(option)]


def _town(name: str) -> str:
    return name.replace("_", " ").title()


def available_values(subject: ConsentSubject) -> Mapping[str, object]:
    """The values the system holds for the policy's fields. A field it does not hold is left out,
    so the consent screen shows it as missing and a release is refused (never a made-up value)."""
    goal, option = subject.goal, subject.option
    values: dict[str, object] = {
        "origin_town": goal.origin_town,
        "destination_town": _town(subject.service.town),
        "specialty": goal.specialty,
        "number_of_travellers": option.travellers,
        "preferred_appointment_slot": option.appointment_start,
    }
    if option.outbound_date and option.return_date:
        values["travel_dates"] = f"{option.outbound_date} to {option.return_date}"
    if option.accommodation_id and option.outbound_date and option.nights:
        values["check_in_date"] = option.outbound_date
        values["check_out_date"] = option.outbound_date + timedelta(days=option.nights)
    contact = subject.contact
    if contact is not None:
        values["given_name"] = contact.given_name
        values["family_name"] = contact.family_name
        values["contact_phone"] = contact.contact_phone
        values["date_of_birth"] = contact.date_of_birth
        values["mobility_assistance_needed"] = contact.mobility_assistance_needed
    return values


def prompt_for(action: str, subject: ConsentSubject, policy: ConsentPolicy) -> ConsentPrompt:
    """The consent screen for one action: the minimum fields with their values."""
    return describe_action(action, policy, available_values(subject))


def release_to_sink(
    request: ConsentRequest,
    subject: ConsentSubject,
    approval_id: str,
    policy: ConsentPolicy,
    log: SimulatedActionLog,
) -> ReleaseRecord:
    """Release the action's fields to its SIMULATED recipient, only under an active consent and the
    id of a recorded approval. All of the policy's fields or nothing; nothing is recorded on a
    refusal."""
    release = release_data(request, available_values(subject), policy)
    return log.record_release(release, approval_id)


def withdraw_with_sink(
    request: ConsentRequest, at: datetime, policy: ConsentPolicy, log: SimulatedActionLog
) -> WithdrawalResult:
    """Withdraw an approved consent and undo its action in the SIMULATED log. The consent is
    checked first, so a request that cannot be withdrawn leaves the log untouched."""
    withdrawn, notice = withdraw_consent(request, at, policy)
    cancelled = log.withdraw(request.action)
    return WithdrawalResult(request=withdrawn, notice=notice, cancelled=cancelled)
