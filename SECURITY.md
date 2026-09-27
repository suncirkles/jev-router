# Security policy

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/suncirkles/jev-router/security/advisories/new),
not in public issues.

This is a research repository without releases. Fixes land on `main` only.

Never commit API keys. `OPENROUTER_API_KEY` and any Modal or Hugging Face credentials belong in
your environment, not in tracked files. Secret scanning with push protection is enabled on this
repository.
