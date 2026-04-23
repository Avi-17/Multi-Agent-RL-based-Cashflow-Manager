# =========================
# Cashflowmanager Environment Client
# =========================

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CashflowmanagerAction, CashflowmanagerObservation, Invoice, Receivable


class CashflowmanagerEnv(
    EnvClient[CashflowmanagerAction, CashflowmanagerObservation, State]
):
    """
    Client for Multi-Agent Cashflow RL Environment.

    Supports:
    - reset()
    - step()
    - state()
    """

    def _step_payload(self, action: CashflowmanagerAction) -> Dict:
        return {
            "type": action.type,
            "invoice_id": action.invoice_id,
            "amount": action.amount,
            "memo": action.memo,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CashflowmanagerObservation]:
        obs_data = payload.get("observation", {})

        invoices = [Invoice(**i) if isinstance(i, dict) else i for i in obs_data.get("invoices", [])]
        receivables = [Receivable(**r) if isinstance(r, dict) else r for r in obs_data.get("receivables", [])]

        observation = CashflowmanagerObservation(
            day=obs_data.get("day", 0),
            cash=obs_data.get("cash", 0.0),
            credit_used=obs_data.get("credit_used", 0.0),
            credit_limit=obs_data.get("credit_limit", 5000.0),
            invoices=invoices,
            receivables=receivables,
            vendor_profiles=obs_data.get("vendor_profiles", {}),
            advisor_memos=obs_data.get("advisor_memos", {}),
            advisor_messages=obs_data.get("advisor_messages", {}),
            world_events=obs_data.get("world_events", []),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )