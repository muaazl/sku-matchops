"""
SKU MatchOps Engine - Standalone Inference & Processing Microservice
Hosts the gRPC inference server (port 50051) and background Redis queue worker.
All Gateway <-> Engine communication runs strictly over gRPC.
"""

import logging
import signal

from engine.core.observability import configure_logging
from engine.grpc.inference_servicer import start_grpc_server
from engine.pipelines.runtime_pipeline import trigger_model_load
from engine.pipelines.worker_pipeline import shutdown_workers

logger = logging.getLogger("matchops.engine.server")
configure_logging(ensure_handlers=True)


def serve(port: int | None = None, host: str = "0.0.0.0") -> None:
    """Starts the gRPC inference server, initiates background model warmup, and blocks until SIGINT/SIGTERM."""
    grpc_server = start_grpc_server(port=port, host=host)
    trigger_model_load(reload=False)

    def _shutdown(signum, _frame):
        logger.info(
            f"[ENGINE] Received signal {signum}; releasing active queue leases & stopping gRPC server..."
        )
        shutdown_workers()
        grpc_server.stop(grace=5)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    serve()
