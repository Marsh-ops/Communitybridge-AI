"""Schema for the Consent Agent's data-field policy (data/policy/data_fields.yaml).

The Consent Agent's only data source is this policy. For each action that would
release personal data it defines the minimum fields, who receives them and why. Data
minimisation is enforced here, when the policy loads, so a careless edit to the YAML fails
immediately instead of quietly over-sharing at run time:
- every field an action uses must be in the catalogue;
- `never_share` fields can't be used by any action;
- `sensitive` fields (e.g. Indigenous status) may only go to an authorised officer (PATS §8.3),
  never to a provider, in line with the Indigenous data governance principles cited in the project plan;
- every action says what withdrawing consent undoes (`on_withdrawal`,).
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from communitybridge.models.common import NonEmptyStr, StrictModel

Sensitivity = Literal["standard", "sensitive", "never_share"]
RecipientType = Literal["simulated_provider", "authorised_officer"]


class DataField(StrictModel):
    description: NonEmptyStr
    sensitivity: Sensitivity


class ConsentAction(StrictModel):
    description: NonEmptyStr
    recipient: NonEmptyStr
    recipient_type: RecipientType
    purpose: NonEmptyStr  # shown to the user on the consent screen
    required_fields: list[NonEmptyStr] = Field(min_length=1)
    # Literal[True]: an action that releases data without approval cannot be defined at all.
    approval_required: Literal[True]
    # Shown next to the withdraw button: what withdrawing undoes for this action.
    # Required, so every action that can be approved can also be withdrawn.
    on_withdrawal: NonEmptyStr


class ConsentPolicy(StrictModel):
    policy_version: NonEmptyStr
    fields: dict[str, DataField] = Field(min_length=1)
    actions: dict[str, ConsentAction] = Field(min_length=1)

    @model_validator(mode="after")
    def _enforce_minimisation(self) -> Self:
        for action_id, action in self.actions.items():
            names = action.required_fields
            if len(names) != len(set(names)):
                raise ValueError(f"{action_id}: duplicate fields in required_fields")
            for name in names:
                field = self.fields.get(name)
                if field is None:
                    raise ValueError(f"{action_id}: field '{name}' is not in the catalogue")
                if field.sensitivity == "never_share":
                    raise ValueError(f"{action_id}: '{name}' is never_share and can't be released")
                to_officer = action.recipient_type == "authorised_officer"
                if field.sensitivity == "sensitive" and not to_officer:
                    raise ValueError(
                        f"{action_id}: sensitive field '{name}' may only go to an officer"
                    )
        return self
