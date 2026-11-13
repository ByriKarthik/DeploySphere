"""MongoDB-independent tests for guarded deployment reconciliation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

from bson import ObjectId
import pytest

from backend.services import deployment_coordinator as coordinator


PROJECT_ID = ObjectId("6af466ba7f1e6d2f280a5177")
DEPLOYMENT_ID = ObjectId("6af4a61fa05c38e44da51b55")
CONTAINER_ID = "71bdb0da3226d107156414f19f2494c81afdc958e62ac61de265a183c936031f"
ATTEMPT_ID = "a" * 32
FIXED_NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)


@pytest.fixture
def forbid_docker_runtime(monkeypatch):
    runtime_mocks = {}
    for operation in (
        "create_container",
        "start_container",
        "stop_container",
        "restart_container",
    ):
        runtime_mock = Mock(name=operation)
        monkeypatch.setattr(coordinator.docker_runtime, operation, runtime_mock)
        runtime_mocks[operation] = runtime_mock
    yield
    for runtime_mock in runtime_mocks.values():
        runtime_mock.assert_not_called()


@pytest.fixture
def mock_now(monkeypatch):
    monkeypatch.setattr(coordinator, "_now", lambda: FIXED_NOW)


def _database(find_one_and_update):
    return SimpleNamespace(
        deployments=SimpleNamespace(find_one_and_update=find_one_and_update)
    )


def _record(**overrides):
    record = {
        "_id": DEPLOYMENT_ID,
        "project_id": PROJECT_ID,
        "status": "failed",
        "container_id": CONTAINER_ID,
        "attempt_id": ATTEMPT_ID,
        "updated_at": FIXED_NOW,
        "error": {
            "code": coordinator.RECONCILIATION_ERROR_CODE,
            "message": coordinator.RECONCILIATION_ERROR_MESSAGE,
            "occurred_at": FIXED_NOW,
        },
    }
    record.update(overrides)
    return record


def test_reconciliation_success_uses_exact_cas_filter_and_update(
    forbid_docker_runtime, mock_now
):
    returned = _record()
    collection_call = Mock(return_value=returned)
    database = _database(collection_call)

    result = coordinator.reconcile_unknown_deployment_as_failed(
        database, PROJECT_ID, DEPLOYMENT_ID, CONTAINER_ID, ATTEMPT_ID
    )

    collection_call.assert_called_once_with(
        {
            "_id": DEPLOYMENT_ID,
            "project_id": PROJECT_ID,
            "status": "unknown",
            "container_id": CONTAINER_ID,
            "attempt_id": ATTEMPT_ID,
        },
        {
            "$set": {
                "status": "failed",
                "updated_at": FIXED_NOW,
                "error": {
                    "code": coordinator.RECONCILIATION_ERROR_CODE,
                    "message": coordinator.RECONCILIATION_ERROR_MESSAGE,
                    "occurred_at": FIXED_NOW,
                },
            }
        },
        return_document=coordinator.ReturnDocument.AFTER,
    )
    assert result is returned


def test_reconciliation_zero_match_returns_conflict_without_retry(
    forbid_docker_runtime, mock_now
):
    collection_call = Mock(return_value=None)

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            _database(collection_call),
            PROJECT_ID,
            DEPLOYMENT_ID,
            CONTAINER_ID,
            ATTEMPT_ID,
        )

    assert exc_info.value.code == "deployment_reconciliation_conflict"
    collection_call.assert_called_once()


def test_reconciliation_omitted_attempt_id_is_rejected(
    forbid_docker_runtime, mock_now
):
    collection_call = Mock()

    with pytest.raises(TypeError, match="attempt_id"):
        coordinator.reconcile_unknown_deployment_as_failed(
            _database(collection_call), PROJECT_ID, DEPLOYMENT_ID, CONTAINER_ID
        )

    collection_call.assert_not_called()


@pytest.mark.parametrize("attempt_id", ["", "invalid-attempt-id"])
def test_reconciliation_invalid_attempt_id_is_rejected(
    forbid_docker_runtime, mock_now, attempt_id
):
    collection_call = Mock()

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            _database(collection_call),
            PROJECT_ID,
            DEPLOYMENT_ID,
            CONTAINER_ID,
            attempt_id,
        )

    assert exc_info.value.code == "invalid_attempt_id"
    collection_call.assert_not_called()


def test_reconciliation_mismatched_attempt_id_cannot_match(
    forbid_docker_runtime, mock_now
):
    collection_call = Mock(return_value=None)
    wrong_attempt_id = "b" * 32

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            _database(collection_call),
            PROJECT_ID,
            DEPLOYMENT_ID,
            CONTAINER_ID,
            wrong_attempt_id,
        )

    assert exc_info.value.code == "deployment_reconciliation_conflict"
    assert collection_call.call_args.args[0]["attempt_id"] == wrong_attempt_id
    assert collection_call.call_count == 1


def test_concurrent_reconciliations_only_one_compare_and_set_succeeds(
    forbid_docker_runtime, mock_now
):
    state = {"status": "unknown"}
    lock = Lock()

    def atomic_find_one_and_update(filter_document, update, *, return_document):
        assert filter_document == {
            "_id": DEPLOYMENT_ID,
            "project_id": PROJECT_ID,
            "status": "unknown",
            "container_id": CONTAINER_ID,
            "attempt_id": ATTEMPT_ID,
        }
        assert return_document == coordinator.ReturnDocument.AFTER
        with lock:
            if state["status"] != filter_document["status"]:
                return None
            state.update(update["$set"])
            return _record(status=state["status"], **{
                key: value for key, value in state.items() if key != "status"
            })

    database = _database(atomic_find_one_and_update)

    def reconcile():
        try:
            return coordinator.reconcile_unknown_deployment_as_failed(
                database, PROJECT_ID, DEPLOYMENT_ID, CONTAINER_ID, ATTEMPT_ID
            )
        except coordinator.DeploymentConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reconcile(), range(2)))

    successes = [result for result in results if isinstance(result, dict)]
    conflicts = [
        result
        for result in results
        if isinstance(result, coordinator.DeploymentConflictError)
    ]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].code == "deployment_reconciliation_conflict"
    assert state["status"] == "failed"
