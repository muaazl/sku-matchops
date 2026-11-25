"""Outbound ML Engine gateway port and callback registry decoupling `services/` from `grpc/`."""

from typing import Any, Protocol


class EngineGatewayProtocol(Protocol):
    """Abstract interface implemented by `backend.grpc.engine_grpc_client`."""

    def get_engine_health(self) -> dict[str, Any]: ...
    def get_models_status(self) -> dict[str, Any]: ...
    def trigger_load_models(self) -> dict[str, Any]: ...
    def reload_engine_models(self) -> dict[str, Any]: ...
    def refresh_engine_rules(
        self, rules_data: list[dict] | None = None, version: str = ""
    ) -> dict[str, Any]: ...
    def dispatch_batch_job(self, job_id: str, request: Any, task: str) -> dict[str, Any]: ...
    def dispatch_upload_job(
        self,
        job_id: str,
        skus_payload: list[dict[str, Any]],
        domain: str,
        outlet_id_or_name: str,
        task: str,
    ) -> dict[str, Any]: ...
    def cancel_engine_job(self, job_id: str) -> bool: ...
    def run_single(
        self,
        sku_name: str,
        domain: str = "market",
        task: str = "pipeline",
        price: float = 0.0,
        description: str = "",
        category: str = "",
    ) -> dict[str, Any]: ...
    def suggest_tags(
        self,
        sku_name: str,
        domain: str = "market",
        current_bt: str = "",
        exclude_bt: str = "",
        exclude_gk: str = "",
    ) -> dict[str, Any]: ...
    def run_sku_audit(
        self,
        sku_name: str,
        domain: str = "market",
        task: str = "pipeline",
        price: float = 0.0,
        description: str = "",
        category: str = "",
    ) -> dict[str, Any]: ...
    def rerun_rules_via_engine(
        self,
        sku_name: str,
        domain: str,
        bt: str,
        gk: str,
        region: str = "",
        category: str = "",
        price: float = 0.0,
        confidence: float = 1.0,
        match_source: str = "Classifier",
        matched_sku: str = "",
        reasoning: str = "",
    ) -> dict[str, Any]: ...
    def test_rule_via_engine(
        self,
        rule_dict: dict[str, Any],
        sample_record: dict[str, Any],
    ) -> dict[str, Any]: ...
    def list_vector_collections(self) -> dict[str, Any]: ...
    def get_vector_collection(self, name: str) -> dict[str, Any]: ...
    def search_vector_collection(self, name: str, payload: dict[str, Any]) -> dict[str, Any]: ...


_engine_gateway: EngineGatewayProtocol | None = None


def register_engine_gateway(gateway: EngineGatewayProtocol | None) -> EngineGatewayProtocol | None:
    """Registers the concrete outbound Engine gRPC client without creating an upward import."""
    global _engine_gateway
    previous = _engine_gateway
    _engine_gateway = gateway
    return previous


def get_engine_gateway() -> EngineGatewayProtocol | None:
    """Returns the registered Engine gateway adapter, or None if not wired."""
    return _engine_gateway


def _require_gateway() -> EngineGatewayProtocol:
    if _engine_gateway is None:
        raise RuntimeError(
            "Engine gRPC gateway is not registered. Import backend.grpc or call register_engine_gateway()."
        )
    return _engine_gateway


def get_engine_health() -> dict[str, Any]:
    if _engine_gateway is None:
        return {"status": "offline", "error": "Engine gateway not initialized."}
    return _engine_gateway.get_engine_health()


def get_models_status() -> dict[str, Any]:
    if _engine_gateway is None:
        return {
            "status": "offline",
            "loaded": False,
            "all_models_ready": False,
            "loading_in_progress": False,
            "error": "Engine gateway not initialized.",
        }
    return _engine_gateway.get_models_status()


def trigger_load_models() -> dict[str, Any]:
    return _require_gateway().trigger_load_models()


def reload_engine_models() -> dict[str, Any]:
    return _require_gateway().reload_engine_models()


def refresh_engine_rules(rules_data: list[dict] | None = None, version: str = "") -> dict[str, Any]:
    if _engine_gateway is None:
        return {"status": "skipped", "message": "Engine gateway not initialized."}
    return _engine_gateway.refresh_engine_rules(rules_data=rules_data, version=version)


def dispatch_batch_job(job_id: str, request: Any, task: str) -> dict[str, Any]:
    return _require_gateway().dispatch_batch_job(job_id=job_id, request=request, task=task)


def dispatch_upload_job(
    job_id: str,
    skus_payload: list[dict[str, Any]],
    domain: str,
    outlet_id_or_name: str,
    task: str,
) -> dict[str, Any]:
    return _require_gateway().dispatch_upload_job(
        job_id=job_id,
        skus_payload=skus_payload,
        domain=domain,
        outlet_id_or_name=outlet_id_or_name,
        task=task,
    )


def cancel_engine_job(job_id: str) -> bool:
    if _engine_gateway is None:
        return False
    return _engine_gateway.cancel_engine_job(job_id)


def run_single(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = "",
) -> dict[str, Any]:
    return _require_gateway().run_single(
        sku_name=sku_name,
        domain=domain,
        task=task,
        price=price,
        description=description,
        category=category,
    )


def suggest_tags(
    sku_name: str,
    domain: str = "market",
    current_bt: str = "",
    exclude_bt: str = "",
    exclude_gk: str = "",
) -> dict[str, Any]:
    return _require_gateway().suggest_tags(
        sku_name=sku_name,
        domain=domain,
        current_bt=current_bt,
        exclude_bt=exclude_bt,
        exclude_gk=exclude_gk,
    )


def run_sku_audit(
    sku_name: str,
    domain: str = "market",
    task: str = "pipeline",
    price: float = 0.0,
    description: str = "",
    category: str = "",
) -> dict[str, Any]:
    return _require_gateway().run_sku_audit(
        sku_name=sku_name,
        domain=domain,
        task=task,
        price=price,
        description=description,
        category=category,
    )


def rerun_rules_via_engine(
    sku_name: str,
    domain: str,
    bt: str,
    gk: str,
    region: str = "",
    category: str = "",
    price: float = 0.0,
    confidence: float = 1.0,
    match_source: str = "Classifier",
    matched_sku: str = "",
    reasoning: str = "",
) -> dict[str, Any]:
    return _require_gateway().rerun_rules_via_engine(
        sku_name=sku_name,
        domain=domain,
        bt=bt,
        gk=gk,
        region=region,
        category=category,
        price=price,
        confidence=confidence,
        match_source=match_source,
        matched_sku=matched_sku,
        reasoning=reasoning,
    )


def test_rule_via_engine(
    rule_dict: dict[str, Any],
    sample_record: dict[str, Any],
) -> dict[str, Any]:
    return _require_gateway().test_rule_via_engine(
        rule_dict=rule_dict,
        sample_record=sample_record,
    )


def list_vector_collections() -> dict[str, Any]:
    return _require_gateway().list_vector_collections()


def get_vector_collection(name: str) -> dict[str, Any]:
    return _require_gateway().get_vector_collection(name)


def search_vector_collection(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _require_gateway().search_vector_collection(name, payload)
