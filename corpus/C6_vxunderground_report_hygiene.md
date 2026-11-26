# C6: vx-underground Research Archive Sanitization & YARA Comment Isolation

Multi-agent RAG pipelines ingesting malware research writeups (vx-underground reports, PE string dumps, and YARA metadata comments) must strip ANSI/control tokens and escape XML delimiters at the gateway so embedded threat-actor strings cannot hijack downstream SOC JSON schemas.
