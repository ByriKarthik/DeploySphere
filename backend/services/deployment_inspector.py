"""Read-only startup inspection of active deployment/container state."""

import logging
import re
from collections import Counter

from bson import ObjectId
from pymongo.errors import PyMongoError

from backend.database.initialize import ACTIVE_DEPLOYMENT_STATUSES
from backend.services import docker_runtime


logger = logging.getLogger(__name__)
CONTAINER_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")
ATTEMPT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
INSPECTION_PROJECTION = {
    "_id": 1,
    "project_id": 1,
    "status": 1,
    "container_id": 1,
    "container_name": 1,
    "attempt_id": 1,
    "container_port": 1,
    "allocated_port": 1,
    "deployment_url": 1,
}
SNAPSHOT_FIELDS = tuple(INSPECTION_PROJECTION)
CLASSIFICATIONS = (
    "running_verified",
    "exited_verified",
    "created_verified",
    "container_not_found",
    "docker_unavailable",
    "identity_mismatch",
    "ambiguous_lookup",
    "missing_identity",
    "unsupported_state",
    "state_mismatch",
    "inspection_inconclusive",
    "stale_database_snapshot",
)
VERIFIED_CLASSIFICATIONS = frozenset(
    {"running_verified", "exited_verified", "created_verified"}
)


def _result(
    document: dict,
    classification: str,
    reason: str,
    observation: docker_runtime.StartupContainerObservation | None = None,
) -> dict:
    return {
        "deployment_id": str(document.get("_id", "")),
        "status": document.get("status"),
        "classification": classification,
        "reason": reason,
        "container_id": observation.container_id if observation else document.get("container_id"),
        "container_name": observation.container_name if observation else document.get("container_name"),
        "docker_state": observation.state if observation else None,
    }


def _valid_port(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= 65535
    )


def _snapshot_is_current(database, document: dict) -> bool:
    deployment_id = document.get("_id")
    if not isinstance(deployment_id, ObjectId):
        return False
    current = database.deployments.find_one(
        {"_id": deployment_id},
        projection=INSPECTION_PROJECTION,
    )
    return current is not None and all(
        current.get(field) == document.get(field) for field in SNAPSHOT_FIELDS
    )


def _observation_classification(
    document: dict,
    inspection: docker_runtime.StartupContainerInspection,
) -> tuple[str, str, docker_runtime.StartupContainerObservation | None]:
    error = inspection.error
    if error is not None:
        if error.code in {
            "container_not_found",
            "docker_unavailable",
            "identity_mismatch",
            "ambiguous_lookup",
            "missing_identity",
        }:
            return error.code, error.message, None
        return "inspection_inconclusive", error.message, None

    observation = inspection.observation
    if observation is None:
        return "inspection_inconclusive", "Docker returned no observation.", None
    expected_name = f"deploysphere-{document['_id']}"
    if (
        not observation.identity_verified
        or observation.container_name != expected_name
        or (
            document.get("container_id") is not None
            and observation.container_id != document.get("container_id")
        )
    ):
        return "identity_mismatch", "Docker identity or ownership labels did not match.", observation

    state = observation.state
    status = document.get("status")
    if state == "running":
        if status != "running":
            return "state_mismatch", "Docker is running while the deployment record is not running.", observation
        expected_port = document.get("container_port")
        expected_key = f"{expected_port}/tcp"
        configured = observation.port_bindings
        if (
            not _valid_port(expected_port)
            or not isinstance(configured, dict)
            or list(configured) != [expected_key]
        ):
            return "state_mismatch", "The configured TCP container port does not match the deployment record.", observation
        configured_bindings = configured[expected_key]
        if (
            not isinstance(configured_bindings, list)
            or len(configured_bindings) != 1
            or not isinstance(configured_bindings[0], dict)
            or configured_bindings[0].get("HostIp") != "127.0.0.1"
            or configured_bindings[0].get("HostPort") != ""
        ):
            return "state_mismatch", "Docker is not configured for one dynamic loopback binding.", observation

        network_ports = observation.network_ports
        if (
            not isinstance(network_ports, dict)
            or list(network_ports) != [expected_key]
        ):
            return "state_mismatch", "Docker did not report exactly the expected runtime TCP mapping.", observation
        actual_bindings = network_ports[expected_key]
        if (
            not isinstance(actual_bindings, list)
            or len(actual_bindings) != 1
            or not isinstance(actual_bindings[0], dict)
        ):
            return "state_mismatch", "Docker did not report exactly one runtime host binding.", observation
        binding = actual_bindings[0]
        host_port = binding.get("HostPort")
        if (
            binding.get("HostIp") != "127.0.0.1"
            or not isinstance(host_port, str)
            or not host_port.isascii()
            or not host_port.isdigit()
            or len(host_port) > 5
            or not _valid_port(int(host_port))
        ):
            return "state_mismatch", "The runtime host binding is not a valid loopback port.", observation
        actual_port = int(host_port)
        if (
            not _valid_port(document.get("allocated_port"))
            or actual_port != document.get("allocated_port")
            or document.get("deployment_url")
            != f"http://127.0.0.1:{document.get('allocated_port')}"
        ):
            return "state_mismatch", "The runtime port does not agree with the persisted port and URL.", observation
        return (
            "running_verified",
            "Docker identity and runtime loopback binding are verified; application health is not implied.",
            observation,
        )

    if state == "exited":
        if status in {
            "queued",
            "creating",
            "starting",
            "running",
            "restarting",
            "checking_ready",
        }:
            return "state_mismatch", "Docker is exited while the deployment record indicates another state.", observation
        return "exited_verified", "Docker reports the identity-verified container as exited.", observation

    if state == "created":
        if status in {"queued", "running", "stopping", "restarting", "checking_ready"}:
            return "state_mismatch", "Docker is created while the deployment record indicates another state.", observation
        return "created_verified", "Docker reports the identity-verified container as created.", observation

    return "unsupported_state", "Docker reported an unsupported container state.", observation


def _inspect_one(database, document: dict) -> dict:
    deployment_id = document.get("_id")
    project_id = document.get("project_id")
    expected_name = (
        f"deploysphere-{deployment_id}" if isinstance(deployment_id, ObjectId) else None
    )
    if (
        not isinstance(deployment_id, ObjectId)
        or not isinstance(project_id, ObjectId)
        or not isinstance(document.get("status"), str)
        or not isinstance(document.get("attempt_id"), str)
        or not ATTEMPT_ID_PATTERN.fullmatch(document["attempt_id"])
    ):
        return _result(document, "missing_identity", "Required deployment identity fields are missing or invalid.")
    if document.get("container_name") != expected_name:
        return _result(document, "identity_mismatch", "The stored container name is not deterministic for this deployment.")
    container_id = document.get("container_id")
    if container_id is not None and (
        not isinstance(container_id, str)
        or not CONTAINER_ID_PATTERN.fullmatch(container_id)
    ):
        return _result(document, "missing_identity", "The stored container ID is not a full Docker ID.")

    try:
        inspection = docker_runtime.inspect_container_for_startup(
            container_id=container_id,
            container_name=expected_name,
            project_id=project_id,
            deployment_id=deployment_id,
        )
    except Exception:
        logger.exception(
            "Startup inspection raised unexpectedly for deployment.",
            extra={"deployment_id": str(deployment_id)},
        )
        inspection = docker_runtime.StartupContainerInspection(
            error=docker_runtime.RuntimeError(
                "inspection_inconclusive",
                "The container inspection raised an unexpected error.",
            )
        )

    try:
        if not _snapshot_is_current(database, document):
            return _result(
                document,
                "stale_database_snapshot",
                "The deployment record changed or disappeared during Docker inspection.",
                inspection.observation,
            )
    except PyMongoError:
        return _result(
            document,
            "inspection_inconclusive",
            "MongoDB could not verify the deployment snapshot after Docker inspection.",
            inspection.observation,
        )

    classification, reason, observation = _observation_classification(
        document, inspection
    )
    return _result(document, classification, reason, observation)


def inspect_active_deployments(database) -> dict:
    """Inspect active deployment records without changing MongoDB or Docker."""
    query = {"status": {"$in": list(ACTIVE_DEPLOYMENT_STATUSES)}}
    try:
        documents = list(
            database.deployments.find(
                query,
                projection=INSPECTION_PROJECTION,
            )
        )
    except PyMongoError as error:
        summary = {
            "candidate_count": 0,
            "classification_counts": {},
            "results": [],
            "database_error": type(error).__name__,
            "outcome": "unavailable",
        }
        logger.error(
            "Startup deployment inspection could not retrieve MongoDB records; continuing without inspection.",
            extra={"startup_inspection_summary": summary},
        )
        return summary

    results = []
    for document in documents:
        try:
            result = _inspect_one(database, document)
        except Exception:
            logger.exception(
                "Startup deployment inspection failed for one record.",
                extra={"deployment_id": str(document.get("_id", ""))},
            )
            result = _result(
                document,
                "inspection_inconclusive",
                "Inspection failed unexpectedly; no state was changed.",
            )
        results.append(result)
        if result["classification"] in {
            "docker_unavailable",
            "inspection_inconclusive",
        }:
            logger.warning(
                "Startup deployment inspection is unresolved.",
                extra={"deployment_inspection": result},
            )

    counts = Counter(result["classification"] for result in results)
    summary = {
        "candidate_count": len(results),
        "classification_counts": {
            classification: counts.get(classification, 0)
            for classification in CLASSIFICATIONS
            if counts.get(classification, 0)
        },
        "results": results,
        "database_error": None,
        "outcome": (
            "completed"
            if all(
                result["classification"] in VERIFIED_CLASSIFICATIONS
                for result in results
            )
            else "partial"
        ),
    }
    logger.info(
        "Startup deployment inspection completed.",
        extra={"startup_inspection_summary": summary},
    )
    return summary
