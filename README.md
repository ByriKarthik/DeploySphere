# DeploySphere

## Backend

Run the local backend with:

```powershell
.\run_backend.ps1
```

The script uses the repository's `.venv`, binds to `127.0.0.1:8001`, and starts
one Uvicorn worker without reload. DeploySphere v1 supports one backend process.
Do not start additional workers or use Uvicorn reload.

All v1 backend instances must run under the same OS user and use the same local
process-lock path. The lock is stored at
`%LOCALAPPDATA%\DeploySphere\backend.lock` on Windows and at
`$XDG_STATE_HOME\deploysphere\backend.lock` on Linux/Unix, falling back to
`~/.local/state/deploysphere/backend.lock`. The lock file remains after shutdown;
only the OS-managed lock determines whether a backend process owns it.
