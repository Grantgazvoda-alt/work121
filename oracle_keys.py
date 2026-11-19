"""Load the domain-separated HMAC oracle key from the environment or a key file.

The eval transport never embeds a key. ``AGENT_ORACLE_KEY`` is hex in the
environment. ``AGENT_ORACLE_KEY_FILE`` points at a hex file that is not
group- or world-readable. Benchmark keys from the parser suite are refused.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_KEY = "AGENT_ORACLE_KEY"
ENV_KEY_FILE = "AGENT_ORACLE_KEY_FILE"
MIN_KEY_BYTES = 16

# Parser-suite fixtures. They stay in that suite's own vectors and cannot be
# reused as the eval oracle key.
_REFUSED_KEYS = frozenset(
    {
        b"benchmark-oracle-key-v1",
        b"wrong-oracle-key-0000000",
    }
)

_HEX_RE = re.compile(r"^(?:0x)?([0-9a-fA-F]+)$")


class OracleKeyError(RuntimeError):
    """The oracle key is missing, malformed, or not safe to use."""


def oracle_key_source() -> str | None:
    """Name of the configured key source, without revealing the key material."""
    if os.environ.get(ENV_KEY):
        return ENV_KEY
    if os.environ.get(ENV_KEY_FILE):
        return ENV_KEY_FILE
    return None


def load_oracle_key(*, material: str | None = None, file_path: str | Path | None = None) -> bytes:
    """Return the oracle key.

    Precedence is explicit ``material``, then ``AGENT_ORACLE_KEY``, then
    ``file_path`` or ``AGENT_ORACLE_KEY_FILE``.
    """
    if material is not None:
        return _validate(_decode_hex(material))

    env_value = os.environ.get(ENV_KEY)
    if env_value:
        return _validate(_decode_hex(env_value))

    path_value = file_path if file_path is not None else os.environ.get(ENV_KEY_FILE)
    if path_value:
        path = Path(path_value)
        _assert_private_file(path)
        return _validate(_decode_hex(path.read_text(encoding="utf-8")))

    raise OracleKeyError(
        f"Set {ENV_KEY} to a hex-encoded key of at least {MIN_KEY_BYTES} bytes, "
        f"or set {ENV_KEY_FILE} to a mode-0600 file containing that hex key."
    )


def _decode_hex(material: str) -> bytes:
    text = "".join(material.split())
    match = _HEX_RE.fullmatch(text)
    if match is None:
        raise OracleKeyError("Oracle key must be hex (optional 0x prefix), with no other characters.")
    hex_body = match.group(1)
    if len(hex_body) % 2 != 0:
        raise OracleKeyError("Oracle key hex must contain an even number of digits.")
    try:
        return bytes.fromhex(hex_body)
    except ValueError as exc:
        raise OracleKeyError("Oracle key hex could not be decoded.") from exc


def validate_oracle_key(key: bytes) -> bytes:
    """Reject short keys and parser-suite benchmark fixtures."""
    if not isinstance(key, (bytes, bytearray)):
        raise OracleKeyError("Oracle key must be bytes.")
    return _validate(bytes(key))


def _validate(key: bytes) -> bytes:
    if len(key) < MIN_KEY_BYTES:
        raise OracleKeyError(f"Oracle key must be at least {MIN_KEY_BYTES} bytes; got {len(key)}.")
    if key in _REFUSED_KEYS:
        raise OracleKeyError(
            "Refusing a parser-suite benchmark key. Generate a unique key and "
            f"place it in {ENV_KEY} or a private {ENV_KEY_FILE}."
        )
    return key


def _assert_private_file(path: Path) -> None:
    if not path.is_file():
        raise OracleKeyError(f"Oracle key file does not exist: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise OracleKeyError(
            f"Oracle key file {path} is group or world accessible (mode {mode:03o}). "
            "chmod 600 the file and retry."
        )
