# Local Oracle development database

Created from the machine's existing native ARM image `gvenzl/oracle-free:23`;
no image was downloaded. The running server reports Oracle AI Database 26ai Free,
release 23.26.2.0.0. This isolated instance does not use OpenSpeech or another
application's database/container.

| Setting | Value |
|---|---|
| Container | `memorizz-observability-oracle` |
| Host / port | `127.0.0.1:1529` (loopback only) |
| Service | `FREEPDB1` |
| Development schema | `MEMORIZZ_DEV` |
| Disposable integration schema | `MEMORIZZ_OBS_TEST_LOCAL` |
| Credentials | `../memorizz-oracle-local/connection.json`, relative to repository root |
| Persistent data | `../memorizz-oracle-local/oradata` |
| Limits | 2 CPUs, 3 GiB container memory, 1 GiB shared memory |

The credentials file is mode 0600 inside a 0700 private directory. Passwords
were randomly generated and are not stored in this repository. Use the
`dev_user`, `dev_password` and `dsn` fields in your database client.
Do not use the test schema for development; the test runner requires it empty
and removes its synthetic tables.

From the repository:

```bash
PYTHONPATH=src .venv/bin/python examples/observability/local_oracle.py check
PYTHONPATH=src .venv/bin/python examples/observability/local_oracle.py test
```

The four live observability tests passed, including concurrent retries,
interrupted-write recovery, tenant isolation and retention. Live verification
found and fixed a first-install Oracle migration error: the state-table MERGE
must be dynamic SQL because its table is created dynamically in the same block.
The test fixture now uses a bounded connection pool, as the production provider
does, instead of rapidly creating dedicated sessions for every operation.

The instance remains running. To stop or restart only this instance:

```bash
docker stop memorizz-observability-oracle
docker start memorizz-observability-oracle
```

It does not auto-start with Docker (`--restart=no`). Host storage is intentional:
Docker's internal disk was full. Only this new instance's failed initial
container/incomplete volume were removed; its successful database, audit and
diagnostic files were retained on private host storage. No existing application
container or data was removed. Free Docker disk space separately before creating
other disk-heavy containers; this setup did not prune images/volumes.

The image's environment and persistence settings follow the
[image owner's documentation](https://github.com/gvenzl/oci-oracle-free).
This is a local development instance, not a production deployment certification.
