# Work item JR-001: matched baseline pilot

Scope: standalone JevRoute library/CLI and independent replay harness; small matched coding subset versus RouteLLM MF and fixed/rule controls. No product integration, cloud deployment, or extra levers.

Acceptance: reproducible data and split manifests; no outcome leakage; real Jev decisions; published RouteLLM checkpoint with formula-verified NumPy inference; immutable run artifacts; tests for policy, quality acceptance, accounting, and resume; report distinguishes archived test success from independently assessed quality.

Budget: at most USD 0.50 in new API usage for the pilot. Existing account key quota is not changed. Historical solver cost is simulated, not newly spent.

Tracking: local work item only; no issue tracker was used.

Status: complete. The frozen 20-task calibration and 40-task evaluation pilot ran successfully; 22 checks and isolated wheel smoke tests passed. See `evidence/pilot-001/REPORT.md`.
