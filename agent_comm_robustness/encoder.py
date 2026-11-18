from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator

from .payloads import PayloadVariant
from .protocol import FramingMode


@dataclass(frozen=True)
class EncodedMessage:
    """Raw byte stream emitted for a PayloadVariant, with integrity metadata."""

    variant: PayloadVariant
    framing: FramingMode
    raw_bytes: bytes
    byte_length: int
    sha256: str


class MessageEncoder:
    """Encodes PayloadVariants into raw wire byte streams under a chosen framing mode."""

    def __init__(self, framing: FramingMode = FramingMode.RAW) -> None:
        self.framing = framing

    def _serialize_body(self, variant: PayloadVariant) -> bytes:
        mut = variant.wire_mutation
        if mut.kind == "raw_bytes":
            return bytes(mut.params["data"])

        if isinstance(variant.payload, dict):
            # allow_nan=True here only because non-finite values are tested as raw strings;
            # dict payloads are serialized canonically without ASCII-escaping valid Unicode.
            body = json.dumps(
                variant.payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        elif isinstance(variant.payload, str):
            body = variant.payload.encode("utf-8")
        elif variant.payload is None:
            body = b""
        else:
            raise TypeError(f"Unsupported payload type: {type(variant.payload).__name__}")

        if mut.kind == "truncate_bytes":
            length = int(mut.params["length"])
            body = body[: max(0, length)]
        elif mut.kind == "prepend_bytes":
            body = bytes(mut.params["data"]) + body
        elif mut.kind == "append_bytes":
            body = body + bytes(mut.params["data"])
        elif mut.kind == "replace_bytes":
            target = bytes(mut.params["target"])
            replacement = bytes(mut.params["replacement"])
            if target not in body:
                raise ValueError(
                    f"replace_bytes target {target!r} not found in case {variant.case_id!r}"
                )
            body = body.replace(target, replacement, 1)

        return body

    def _apply_framing(self, body: bytes, variant: PayloadVariant) -> bytes:
        mut = variant.wire_mutation
        if mut.bypass_framing or self.framing == FramingMode.RAW:
            return body

        if self.framing == FramingMode.LENGTH_PREFIXED:
            if mut.kind == "corrupt_frame_length":
                if "declared_len" in mut.params:
                    declared = int(mut.params["declared_len"])
                else:
                    declared = max(0, len(body) + int(mut.params.get("delta", 0)))
                return struct.pack(">I", declared) + body
            return struct.pack(">I", len(body)) + body

        if self.framing == FramingMode.NDJSON:
            if mut.kind == "strip_ndjson_newline":
                return body
            if mut.kind == "inject_ndjson_newline":
                extra_line = bytes(mut.params.get("extra_line", b"{}"))
                return body + b"\n" + extra_line + b"\n"
            return body + b"\n"

        raise ValueError(f"Unsupported framing mode: {self.framing}")

    def encode(self, variant: PayloadVariant) -> EncodedMessage:
        body = self._serialize_body(variant)
        wire = self._apply_framing(body, variant)
        digest = hashlib.sha256(wire).hexdigest()
        return EncodedMessage(
            variant=variant,
            framing=self.framing,
            raw_bytes=wire,
            byte_length=len(wire),
            sha256=digest,
        )

    def encode_stream(self, variants: Iterable[PayloadVariant]) -> Iterator[EncodedMessage]:
        for variant in variants:
            yield self.encode(variant)
