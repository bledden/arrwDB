# Claude Code Rules

## Destructive Actions Require Confirmation

NEVER perform any of the following without explicitly asking the user first and receiving confirmation:

- Resetting, stopping, or deleting VMs or cloud infrastructure
- Running `killall`, `kill -9`, or broad process termination commands on remote machines
- Deleting databases, data files, checkpoints, or cached results
- Force-pushing git branches
- Any action that could lose hours of compute work or data

When SSH or connectivity issues arise, ask the user to verify from their end before taking destructive recovery steps.

## No Dummy/Mock Values

NEVER use dummy, mock, fake, or placeholder values for API keys, credentials, secrets, or configuration without explicit instruction from the user. Always use real values or ask the user to provide them.
