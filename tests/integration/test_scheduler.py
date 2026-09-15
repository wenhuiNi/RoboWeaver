import pytest

from roboweaver.adapters.mock import MockExecutor
from roboweaver.contracts import ActionProposal, ActionStatus, Verdict, VerificationResult
from roboweaver.ledger import Ledger
from roboweaver.scheduler import Scheduler

pytestmark = pytest.mark.integration


def proposal(s, run, value=True):
    return ActionProposal(
        action_type="set_flag",
        arguments={"value": value},
        observation_id=s.ledger.read(run).observation.observation_id,
    )


async def test_i03_capacity_sequence_and_explicit_end(rig):
    s, e, c, run = rig
    assert s.ledger.read(run).status != "SUCCEEDED"
    assert s.append(run, 1, [proposal(s, run), proposal(s, run)]) == 2
    with pytest.raises(ValueError, match="capacity"):
        s.append(run, 1, [proposal(s, run)])
    with pytest.raises(ValueError, match="final sequence"):
        s.close(run, 1, 3)
    s.close(run, 1, 2)
    await s.dispatch(run)
    assert e.starts == 1
    assert await s.dispatch(run) is None
    assert not s.ledger.read(run).status == "SUCCEEDED"


async def test_i04_revision_waits_for_stop(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run), proposal(s, run)])
    await s.dispatch(run)
    e.confirm_cancel = False
    await s.request_stop(run, "replan")
    with pytest.raises(ValueError, match="not confirmed"):
        s.revise(run, 1, await e.observe())
    assert await s.dispatch(run) is None
    a = s.ledger.read(run).actions[0]
    s.apply(e.inject(a.idempotency_key, ActionStatus.CANCELED))
    assert s.revise(run, 1, await e.observe()) == 2
    with pytest.raises(ValueError, match="version"):
        s.append(run, 1, [proposal(s, run)])
    s.append(run, 2, [proposal(s, run)])
    await s.dispatch(run)
    assert e.starts == 2
    assert s.ledger.read(run).actions[1].status == "INVALIDATED"


async def test_i05_duplicate_out_of_order_old_event(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    await s.dispatch(run)
    a = s.ledger.read(run).actions[0]
    event = e.inject(a.idempotency_key)
    s.apply(event)
    s.apply(event)
    late = event.model_copy(
        update={
            "event_id": "late",
            "status": ActionStatus.RUNNING,
            "sequence": 20,
            "stopped": False,
        }
    )
    s.apply(late)
    assert s.ledger.read(run).actions[0].status == "COMPLETED"
    await s.request_stop(run, "replan")
    s.revise(run, 1, await e.observe())
    s.apply(late)
    assert s.ledger.read(run).status == "READY"


async def test_i06_lost_reply_and_reconcile(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    e.lose_next_reply = True
    await s.dispatch(run)
    assert s.ledger.read(run).status == "BLOCKED"
    await s.dispatch(run)
    assert e.starts == 1
    await s.reconcile(run)
    assert s.ledger.read(run).actions[0].execution_id
    assert e.starts == 1


async def test_i10_reopen_ledger_without_resubmitting(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    e.lose_next_reply = True
    await s.dispatch(run)
    restored = Scheduler(Ledger(s.ledger.path), MockExecutor(e.path, c), c)
    await restored.reconcile(run)
    await restored.dispatch(run)
    assert restored.ledger.read(run).actions[0].status == "RUNNING"
    assert e.starts == 1


async def test_i03_checkpoint_gates_next_action(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run), proposal(s, run)])
    await s.dispatch(run)
    a = s.ledger.read(run).actions[0]
    s.apply(e.inject(a.idempotency_key))
    assert await s.dispatch(run) is None
    obs = await e.observe()
    s.observe(run, obs)
    s.verify(
        run,
        VerificationResult(
            verdict=Verdict.PASS,
            checkpoint=a.action_id,
            observation_id=obs.observation_id,
            observed_at=obs.observed_at,
            reason="fixture",
            source="mock",
        ),
    )
    await s.dispatch(run)
    assert e.starts == 2
