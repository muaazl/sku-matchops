"""
Network security utilities for Engine worker processes.
Protects against Server-Side Request Forgery (SSRF) and malicious outbound calls.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger("matchops.engine.security")

DISALLOWED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_outbound_url(url: str, allowed_domains: tuple[str, ...] | None = None) -> str:
    """Validates an outbound URL to prevent Server-Side Request Forgery (SSRF).

    Rejects:
      - Non-http/https schemes
      - Missing hostname
      - Domains not in allowed_domains (if provided)
      - Resolution to private, loopback, or link-local IP addresses (RFC 1918 / 3927)
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    hostname_lower = hostname.lower()

    if allowed_domains is not None:
        if not any(
            hostname_lower == d.lower() or hostname_lower.endswith("." + d.lower())
            for d in allowed_domains
        ):
            raise ValueError(
                f"Domain '{hostname}' is not in the allowed outbound whitelist: {allowed_domains}"
            )

    if hostname_lower in ("localhost", "metadata.google.internal", "instance-data"):
        raise ValueError(f"Access to internal hostname '{hostname}' is prohibited.")

    try:
        try:
            ip_addrs = [ipaddress.ip_address(hostname)]
        except ValueError:
            addr_info = socket.getaddrinfo(hostname, None)
            ip_addrs = [ipaddress.ip_address(sockaddr[0]) for *_, sockaddr in addr_info]

        for ip_addr in ip_addrs:
            for net in DISALLOWED_NETWORKS:
                if ip_addr in net:
                    raise ValueError(
                        f"Access to private/internal network address '{ip_addr}' is prohibited."
                    )
    except socket.gaierror as e:
        logger.warning(f"Could not resolve hostname '{hostname}' during pre-flight SSRF check: {e}")

    return url
