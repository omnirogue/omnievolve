# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/omnirogue/omnievolve/security/advisories/new)
for this repository. Do not open a public issue. Include the version, a description, and steps
to reproduce. We aim to acknowledge reports within three business days.

## Supported versions

Security fixes go to the latest release.

## Threat model

OmniEvolve executes code by design:

- A **subprocess evaluator** runs the command in your config for every candidate, and the
  candidate itself is a program the search wrote (possibly with a language model). Treat
  candidates as untrusted. Use timeouts, and run in a container or VM when the benchmark has
  access to anything valuable.
- A **Python callable evaluator** runs in the OmniEvolve process with no isolation and no
  timeout. Use it only for trusted, fast scoring code.
- The **MCP server** runs evaluators on behalf of an AI client. It resolves every path inside
  `--root` and refuses anything outside it; give it the narrowest root that works.
- **API keys** are read from environment variables at call time and are never written to the
  ledger, the HTML report or error messages.
- **`--cloud`** uploads the target, settings and the project's text files to the endpoint in
  `OMNISI_CLOUD_URL` (HTTPS only, except `localhost`). Secret-bearing file names are always
  skipped; review the payload with `--dry-run` first.
- **HTML reports** escape all candidate text and load no external resources or scripts.
