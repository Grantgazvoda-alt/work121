# C7: Prompt Injection & Jailbreak Defense Reference (MITRE ATLAS & OWASP)

Defensive frameworks map indirect prompt injection and jailbreak vectors across OWASP LLM01:2025, OWASP Agentic ASI01:2026 (Goal Hijacking), MITRE ATLAS (AML.T0051, AML.T0054), and NIST AI 100-2e2025: defenses combine Unicode tag/zero-width stripping (U+E0000..U+E007F), special token neutralization, markdown image-exfiltration blocking, Policy Puppetry schema validation, and CaMeL-style instruction-data isolation.
