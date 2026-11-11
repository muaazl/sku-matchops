import unittest

from pydantic import ValidationError

from backend.app.schemas.models import RuleModel


class TestSecurityHardening(unittest.TestCase):
    def test_pydantic_rule_validation_valid(self):
        # Valid rule schema should not raise validation errors
        valid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": "A valid rule description.",
            "reasoning": "Reasoning behind this rule.",
            "condition_logic": "AND",
            "is_active": 1,
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "sku_contains",
                    "value": "coke",
                    "negate": 0,
                }
            ],
            "actions": [{"action_type": "set_bt", "value": "Soft Drink"}],
        }
        try:
            RuleModel(**valid_payload)
        except ValidationError as e:
            self.fail(f"ValidationError raised unexpectedly for valid payload: {e}")

    def test_pydantic_rule_validation_invalid_domain(self):
        # Invalid domain name should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "invalid_domain",  # Rejected
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_excessive_lengths(self):
        # Description/Reasoning fields exceeding length limits should be rejected
        long_description = "A" * 251  # Max limit is 250
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": long_description,
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_condition_type(self):
        # Invalid condition type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "invalid_type",  # Rejected
                    "value": "test",
                }
            ],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_action_type(self):
        # Invalid action type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [
                {
                    "action_type": "invalid_action",  # Rejected
                    "value": "test",
                }
            ],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def tearDown(self):
        super().tearDown()
        from backend.app.core import config
        from conftest import TEST_DATABASE_URL

        config.AUTH_ENABLED = False
        config.ENVIRONMENT = "development"
        config.JWT_SECRET = ""
        config.INTERNAL_GRPC_SECRET = ""
        config.DATABASE_URL = TEST_DATABASE_URL

    def test_config_rejects_empty_api_key(self):
        """SEC-01: Empty MATCHOPS_API_KEY must raise RuntimeError."""
        import importlib
        import os
        from unittest.mock import patch

        try:
            with patch.dict(
                os.environ,
                {
                    "MATCHOPS_API_KEY": "",
                },
            ):
                from backend.app.core import config

                with self.assertRaises(RuntimeError) as ctx:
                    importlib.reload(config)
                self.assertIn("non-empty MATCHOPS_API_KEY", str(ctx.exception))
        finally:
            from backend.app.core import config
            from conftest import TEST_DATABASE_URL

            config.AUTH_ENABLED = False
            config.ENVIRONMENT = "production"
            config.JWT_SECRET = ""
            config.INTERNAL_GRPC_SECRET = ""
            config.DATABASE_URL = TEST_DATABASE_URL

    def test_api_key_empty_rejected(self):
        """SEC-01: Empty or unset config.API_KEY must reject incoming API key even if empty."""
        from unittest.mock import MagicMock, patch

        from fastapi import HTTPException
        from starlette.requests import Request

        from backend.app.core import config
        from backend.app.core.security import get_principal

        mock_req = MagicMock(spec=Request)
        with patch.object(config, "API_KEY", ""):
            with self.assertRaises(HTTPException) as ctx:
                get_principal(request=mock_req, bearer=None, api_key="")
            self.assertEqual(ctx.exception.status_code, 401)

    def test_grpc_interceptor_validates_secret(self):
        """SEC-04: GrpcContextInterceptor must enforce and reject invalid/missing secrets."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import grpc

        from backend.app.core.grpc_utils import GrpcContextInterceptor

        interceptor = GrpcContextInterceptor()

        def dummy_handler(request, context):
            return "ok"

        def continuation(details):
            return SimpleNamespace(
                unary_unary=dummy_handler,
                request_deserializer=None,
                response_serializer=None,
            )

        with patch("backend.app.core.config.INTERNAL_GRPC_SECRET", "super_secret_token"):
            # 1. Missing secret header -> Aborts with UNAUTHENTICATED
            details_missing = SimpleNamespace(
                invocation_metadata=[("x-request-id", "trace-1")],
                unary_unary=dummy_handler,
            )
            handler_missing = interceptor.intercept_service(continuation, details_missing)
            context_mock = MagicMock()
            handler_missing.unary_unary(None, context_mock)
            context_mock.abort.assert_called_once_with(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing internal gRPC secret."
            )

            # 2. Invalid secret header -> Aborts with UNAUTHENTICATED
            details_invalid = SimpleNamespace(
                invocation_metadata=[("x-internal-secret", "wrong_token")],
                unary_unary=dummy_handler,
            )
            handler_invalid = interceptor.intercept_service(continuation, details_invalid)
            context_mock_invalid = MagicMock()
            handler_invalid.unary_unary(None, context_mock_invalid)
            context_mock_invalid.abort.assert_called_once_with(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing internal gRPC secret."
            )

            # 3. Valid secret header -> Succeeds without abort
            details_valid = SimpleNamespace(
                invocation_metadata=[("x-internal-secret", "super_secret_token")],
                unary_unary=dummy_handler,
            )
            handler_valid = interceptor.intercept_service(continuation, details_valid)
            context_mock_valid = MagicMock()
            res = handler_valid.unary_unary(None, context_mock_valid)
            self.assertEqual(res, "ok")
            context_mock_valid.abort.assert_not_called()

    def test_qdrant_proxy_requires_platform_admin(self):
        """SEC-06: Non-platform-admin callers must be rejected with HTTP 403 Forbidden on /vector-db/."""
        from fastapi import HTTPException

        from backend.app.api.endpoints.qdrant_proxy import router as qdrant_router
        from backend.app.core.security import Principal, require_platform_admin

        # 1. Verify route dependency registration on APIRouter
        self.assertTrue(
            any(d.dependency == require_platform_admin for d in qdrant_router.dependencies)
        )

        # 2. Non-admin is rejected with 403
        with self.assertRaises(HTTPException) as ctx:
            require_platform_admin(Principal(username="user", role="reviewer"))
        self.assertEqual(ctx.exception.status_code, 403)

        with self.assertRaises(HTTPException) as ctx:
            require_platform_admin(Principal(username="eng", role="rule_engineer"))
        self.assertEqual(ctx.exception.status_code, 403)

        # 3. Admin is allowed
        p_admin = Principal(username="admin", role="admin")
        self.assertEqual(require_platform_admin(p_admin), p_admin)

    def test_outbound_ssrf_protection_blocks_internal_and_metadata_ips(self):
        """SEC-02 & SEC-03: validate_outbound_url blocks loopback, private, and metadata IPs."""
        from backend.app.core.security import validate_outbound_url
        from engine.core.security import validate_outbound_url as engine_validate_url

        disallowed_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:5432/query",
            "http://10.0.0.1:8000/internal",
            "http://172.16.0.5:9090/metrics",
            "http://192.168.1.100/admin",
            "http://localhost:50051",
            "ftp://pickme.lk/csv",
        ]

        for u in disallowed_urls:
            with self.assertRaises(ValueError):
                validate_outbound_url(u)
            with self.assertRaises(ValueError):
                engine_validate_url(u)

    def test_portal_service_blocks_ssrf_attempts(self):
        """SEC-02: fetch_merchant_csv rejects SSRF attempts without sending HTTP requests."""
        from unittest.mock import patch

        from backend.app.services.portal_service import fetch_merchant_csv

        # 1. AWS metadata attempt via portal_url
        with patch("requests.get") as mock_get:
            res = fetch_merchant_csv(
                "MER123", "token_xyz", portal_url="http://169.254.169.254/latest/meta-data"
            )
            self.assertIn("Invalid portal URL", res["error"])
            mock_get.assert_not_called()

        # 2. Arbitrary domain attempt outside allowed whitelist
        with patch("requests.get") as mock_get:
            res = fetch_merchant_csv(
                "MER123",
                "token_xyz",
                portal_url="https://attacker.controlled.domain.com/csv/{merchantid}",
            )
            self.assertIn("Invalid portal URL", res["error"])
            mock_get.assert_not_called()
