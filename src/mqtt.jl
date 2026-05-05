# ── MQTT client wrapper ──────────────────────────────────────────────────────

function parse_broker_url(url::String)
    stripped = replace(url, r"^tcp://" => "")
    parts = split(stripped, ":")
    host = String(parts[1])
    port = length(parts) > 1 ? parse(Int, parts[2]) : 1883
    return (host, port)
end

function subscribe_and_process!(config::WorkerConfig)
    host, port = parse_broker_url(config.broker_url)

    params_topic = "compute/jobs/$(config.job_id)/params"
    result_topic = "compute/jobs/$(config.job_id)/result"
    status_topic = "compute/jobs/$(config.job_id)/status"
    log_topic    = "compute/jobs/$(config.job_id)/logs"

    @info "Connecting to MQTT broker" host=host port=port
    @info "Subscribing to" topic=params_topic

    client = Mosquitto.Client(; id=config.worker_id)

    username = something(config.mqtt_username, "")
    password = something(config.mqtt_password, "")
    Mosquitto.connect(client, host, port; username=username, password=password, keepalive=30)
    Mosquitto.subscribe(client, params_topic; qos=1)

    @info "Waiting for job on $(params_topic)..."

    msg_channel = Mosquitto.get_messages_channel(client)
    message = nothing
    while message === nothing
        Mosquitto.loop(client; timeout=500, ntimes=10)
        while !isempty(msg_channel)
            msg = take!(msg_channel)
            if msg.topic == params_topic
                message = msg
                break
            end
        end
    end

    publish_status!(client, status_topic, config, "processing")

    log_fn = (level, msg) -> publish_log!(client, log_topic, config.job_id, level, msg)

    result = process_job(message.payload, config.worker_id, log_fn)

    result_payload = String(serialize_result(result))
    Mosquitto.publish(client, result_topic, result_payload; qos=1)

    status = result.success ? "completed" : "error"
    publish_status!(client, status_topic, config, status)

    @info "Job processing complete" success=result.success
    Mosquitto.disconnect(client)
end

function publish_status!(client, topic::String, config::WorkerConfig, status::String)
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
    payload = JSON3.write(Dict(
        "job_id" => config.job_id,
        "worker_id" => config.worker_id,
        "status" => status,
        "timestamp" => timestamp,
    ))
    Mosquitto.publish(client, topic, payload; qos=0)
end

function publish_log!(client, topic::String, job_id::String, level::String, message::String)
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
    payload = JSON3.write(Dict(
        "job_id"    => job_id,
        "level"     => level,
        "message"   => message,
        "timestamp" => timestamp,
    ))
    Mosquitto.publish(client, topic, payload; qos=0)
end

# ── Entry point ──────────────────────────────────────────────────────────────

function julia_main()::Cint
    try
        @info "========================================"
        @info "  Grassmann Distance Worker (Julia)"
        @info "========================================"

        config = load_config()
        log_config(config)

        subscribe_and_process!(config)

        @info "Worker shutdown complete"
        return 0
    catch e
        @error "Fatal error" exception=(e, catch_backtrace())
        return 1
    end
end
