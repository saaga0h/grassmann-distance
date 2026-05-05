# ── Worker configuration from environment ────────────────────────────────────

struct WorkerConfig
    broker_url::String
    job_id::String
    worker_id::String
    mqtt_username::Union{String, Nothing}
    mqtt_password::Union{String, Nothing}
    use_gpu::Bool
end

function load_config()::WorkerConfig
    job_id = get(ENV, "JOB_ID") do
        error("JOB_ID environment variable is required")
    end

    broker = get(ENV, "MQTT_BROKER", "tcp://localhost:1883")
    worker_id = get(ENV, "WORKER_ID", "grassmann-$(job_id)")

    username = get(ENV, "MQTT_USER", nothing)
    password = get(ENV, "MQTT_PASSWORD", nothing)

    use_gpu = lowercase(get(ENV, "USE_GPU", "false")) in ("true", "1")

    WorkerConfig(broker, job_id, worker_id, username, password, use_gpu)
end

function log_config(config::WorkerConfig)
    @info "Configuration" broker=config.broker_url job_id=config.job_id worker_id=config.worker_id use_gpu=config.use_gpu
end
