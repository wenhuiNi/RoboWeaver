"""Task verifiers return evidence; execution completion is a separate fact."""

from roboweaver.contracts import Observation, TaskSpec, Verdict, VerificationResult


def state_goal(task: TaskSpec, observation: Observation, checkpoint: str) -> VerificationResult:
    """Synthetic fixture verifier: exact state matching, not visual understanding."""
    expected = task.parameters.get("expected_state")
    if not isinstance(expected, dict) or not expected or not observation.valid:
        verdict = Verdict.UNKNOWN
    elif any(key not in observation.state for key in expected):
        verdict = Verdict.UNKNOWN
    else:
        verdict = (
            Verdict.PASS
            if all(observation.state[k] == v for k, v in expected.items())
            else Verdict.FAIL
        )
    return VerificationResult(
        verdict=verdict,
        checkpoint=checkpoint,
        observation_id=observation.observation_id,
        observed_at=observation.observed_at,
        reason="Exact state comparison (synthetic fixture)",
        source=observation.source,
    )
