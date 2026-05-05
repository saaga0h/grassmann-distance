# MQTT Worker

<!-- @tier: 2 -->
<!-- @parent: ARCHITECTURE.md -->
<!-- @source: src/mqtt.jl, src/config.jl, singularity.def, deploy/nomad/grassmann-distance.hcl -->
<!-- @see-also: docs/messaging.md -->

## Overview

The MQTT subsystem is the runtime entry point for all FORGE-dispatched jobs. It owns the worker lifecycle from broker connection through result publication and disconnect. Every production execution of the GrassmannDistance worker passes through this subsystem; no other entry point is used in deployment.

The worker is one-shot: it connects to the MQTT broker, subscribes to a single job-specific params topic, receives exactly one retained message, delegates computation to `src/app.jl`, publishes the result and final status, then disconnects and exits. One Nomad dispatch equals one MQTT session equals one job.

## Key Files & Entry Points

| File | Role |
|---|---|
| `src/mqtt.jl` | All broker interaction: connection, subscribe/receive loop, `publish_status!`, `publish_log!`, and `julia_main` entry point |
| `src/config.jl` | `WorkerConfig` struct and `load_config()` — reads all environment variables before any broker connection is attempted |
| `singularity.def` (`%runscript`) | Container entry point — validates required env vars, resolves sysimage path, then invokes `julia_main` via `GrassmannDistance.julia_main()` |
| `deploy/nomad/grassmann-distance.hcl` | Nomad parameterized batch job definition — sets `JOB_ID`, `WORKER_ID`, `USE_GPU`, and injects MQTT credentials from Vault |

## Architecture

The MQTT subsystem is structured as a thin integration layer. It does not contain any computation logic — all job processing is delegated to `process_job` in `src/app.jl`. The subsystem's responsibilities are:

1. Parsing the broker URL (`parse_broker_url`)
2. Constructing the four job-scoped topic strings from `JOB_ID`
3. Managing the `Mosquitto.Client` lifecycle (connect → subscribe → loop → disconnect)
4. Wrapping computation progress in status and log MQTT messages via `publish_status!` and `publish_log!`
5. Owning `julia_main`, the sole `Cint`-returning function that the container runscript calls via `exit(GrassmannDistance.julia_main())`

The Mosquitto client is configured with a 30-second keepalive and subscribes at QoS 1. Status and log messages are published at QoS 0 (fire-and-forget). The result is published at QoS 1.

**How to add a new status or log message:**

- For a status update (one of `"processing"`, `"completed"`, or `"error"`): call `publish_status!(client, status_topic, config, "<status>")`. The `status_topic` variable is already constructed inside `subscribe_and_process!` as `compute/jobs/$(config.job_id)/status`.
- For a progress or diagnostic log message: call `publish_log!(client, log_topic, config.job_id, "<level>", "<message>")` where `level` is `"info"` or `"error"`. The `log_topic` variable follows the same pattern. In practice, new log calls in `src/app.jl` are made via the `log_fn` closure that `subscribe_and_process!` constructs and passes to `process_job`:
  ```julia
  log_fn = (level, msg) -> publish_log!(client, log_topic, config.job_id, level, msg)
  ```
  Any code path that receives `log_fn` can call it directly without touching `src/mqtt.jl`.

## Worker Lifecycle

```
singularity run worker.sif
  └─ %runscript: validate env, resolve sysimage, exec julia
       └─ GrassmannDistance.julia_main()
            ├─ load_config()           # read env vars into WorkerConfig
            ├─ select_backend(use_gpu) # CPU() or AMDGPU backend
            └─ subscribe_and_process!(config, backend)
                 ├─ parse_broker_url(config.broker_url)
                 ├─ Mosquitto.Client(id=config.worker_id)
                 ├─ Mosquitto.connect(host, port; keepalive=30)
                 ├─ Mosquitto.subscribe(params_topic; qos=1)
                 │
                 │   [poll loop — Mosquitto.loop every 500ms]
                 │
                 ├─ receive retained params message
                 ├─ publish_status!(... "processing")
                 ├─ process_job(payload, worker_id, log_fn; backend=backend)
                 ├─ Mosquitto.publish(result_topic, result_payload; qos=1)
                 ├─ publish_status!(... "completed" | "error")
                 └─ Mosquitto.disconnect(client)
```

The receive loop calls `Mosquitto.loop(client; timeout=500, ntimes=10)` and drains `Mosquitto.get_messages_channel(client)` until a message arrives on the params topic. Any message on a different topic is silently discarded. Because FORGE publishes params as a **retained message before dispatching the worker**, the retained message is delivered immediately after subscription — the loop typically completes in its first iteration.

## MQTT Topics

All topics are scoped to `config.job_id`. The four topic variables constructed in `subscribe_and_process!` are:

| Variable | Topic pattern | Direction | QoS | Retained |
|---|---|---|---|---|
| `params_topic` | `compute/jobs/{job_id}/params` | FORGE → worker | 1 | yes |
| `result_topic` | `compute/jobs/{job_id}/result` | worker → FORGE | 1 | no |
| `status_topic` | `compute/jobs/{job_id}/status` | worker → FORGE | 0 | no |
| `log_topic` | `compute/jobs/{job_id}/logs` | worker → FORGE | 0 | no |

The worker subscribes only to `params_topic`. It never subscribes to wildcards or client-facing topics.

For full message schemas (payload shapes, field definitions, status values, log levels), see [docs/messaging.md](../../messaging.md).

## Configuration

`load_config()` in `src/config.jl` reads the following environment variables and populates a `WorkerConfig` struct. The runscript in `singularity.def` validates that the required variables are non-empty before Julia starts.

| Variable | `WorkerConfig` field | Default | Required | Notes |
|---|---|---|---|---|
| `MQTT_BROKER` | `broker_url` | `tcp://localhost:1883` | yes (production) | Parsed by `parse_broker_url` — strips `tcp://` prefix, splits on `:` for host and port |
| `JOB_ID` | `job_id` | — | yes | Missing value raises an error before any broker connection |
| `WORKER_ID` | `worker_id` | `grassmann-{JOB_ID}` | no | Nomad sets this to `nomad-{NOMAD_ALLOC_ID}` |
| `MQTT_USER` | `mqtt_username` | `nothing` | yes (production) | Passed as empty string to `Mosquitto.connect` if `nothing` |
| `MQTT_PASSWORD` | `mqtt_password` | `nothing` | yes (production) | Same treatment as `MQTT_USER` |
| `USE_GPU` | `use_gpu` | `false` | no | Accepts `"true"` or `"1"` (case-insensitive) |

In the Nomad job, `MQTT_BROKER`, `MQTT_USER`, and `MQTT_PASSWORD` are injected from Vault path `secret/data/nomad/forge` via the `template` block — they are never baked into the image or the HCL file.

## Error Handling

**Application-level errors** (parse failure, unknown mode, computation exception) are caught inside `process_job` in `src/app.jl`. They result in a `JobResult` with `success=false` and an `error` string. The worker serializes this result and publishes it to `result_topic` at QoS 1, then publishes `"error"` status to `status_topic`, and disconnects normally. Exit code is `0` — the job ran to completion from the worker's perspective.

**Fatal errors** (broker unreachable, `JOB_ID` missing, unexpected panic) are caught by the `try/catch` in `julia_main`:

```julia
function julia_main()::Cint
    try
        ...
        subscribe_and_process!(config, backend)
        return 0
    catch e
        @error "Fatal error" exception=(e, catch_backtrace())
        return 1
    end
end
```

On a fatal error, `julia_main` logs the backtrace to stderr and returns exit code `1`. The container exits with a non-zero status. No result or error status is published to MQTT — FORGE must detect the missing result via timeout or Nomad job failure status.

No restart or reschedule is configured in the Nomad job (`attempts = 0` for both `restart` and `reschedule`). A worker failure is terminal; retry is FORGE's responsibility.

**How to diagnose MQTT problems:**

1. **Check broker connectivity.** Confirm the broker address from `MQTT_BROKER` is reachable from the Nomad allocation's network namespace. The job uses `network.mode = "host"`, so the host's routing applies directly.
2. **Verify the retained params message.** Use any MQTT client to subscribe to `compute/jobs/{job_id}/params` with the correct credentials. The retained message must be present before the worker is dispatched — if it is absent, the worker will block in the poll loop indefinitely and eventually be killed by Nomad's `kill_timeout = "30s"`.
3. **Check Nomad allocation logs.** All Julia log output (`@info`, `@error`) goes to stderr. Retrieve it with:
   ```
   nomad alloc logs -stderr <alloc-id>
   ```
   The startup sequence logs broker host, port, params topic, and config values before any network I/O.

## Dependencies

- `Mosquitto.jl` — Julia binding to `libmosquitto`. Requires `libmosquitto-dev` at build time (installed in the container `%post` section) and the shared library at runtime.
- `JSON3.jl` — used by `publish_status!` and `publish_log!` to serialize status and log payloads.
- `Dates.jl` — provides UTC timestamps in ISO 8601 format (`yyyy-mm-ddTHH:MM:SS.sssZ`) for all published messages.

## Related Documents

- [docs/messaging.md](../../messaging.md) — full MQTT protocol: topic map, message schemas, status values, log levels, graph blob format, and the complete FORGE-to-worker sequencing diagram
- [docs/development.md](../../development.md) — configuration reference, Nomad job behavior, Vault secret injection, container build, and troubleshooting table
- `src/app.jl` — `process_job`, `_process_build`, `_process_query` — everything the worker does between receiving params and publishing the result
- `src/config.jl` — `WorkerConfig` struct and `load_config()`
- `deploy/nomad/grassmann-distance.hcl` — authoritative Nomad job definition
- `singularity.def` — container build and runscript
