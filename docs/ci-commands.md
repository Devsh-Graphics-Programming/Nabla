# CI Commit Commands

Push workflows read simple commands from the commit message body.

| Command | Effect |
| --- | --- |
| `/ci skip` | Do not run CI for this push. The legacy `no CI` and `no ci` markers are still accepted. |
| `/ci keep` | Run CI for this push without cancelling older runs on the same branch. GPU Jenkins jobs still serialize and wait for the runner slot. |

Without a command, push CI uses latest-commit-wins behavior per branch. New pushes cancel older queued or running GitHub Actions runs on the same branch, and the path tracer trigger also asks Jenkins to stop older matching branch runs before starting the new one.
