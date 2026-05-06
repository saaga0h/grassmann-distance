# Grassmann Distance Worker — Nomad Parameterized Batch Job
#
# Dispatched by Forge for each compute request. One dispatch = one job = exits.
# Forge sets job_id in dispatch metadata; worker reads NOMAD_META_job_id to
# subscribe to the right MQTT topic.
#
# The .sif is read directly from /nfs/images — the GPU node has this share mounted.
# No HTTP artifact download: Singularity images are too large for that path.
#
# Deploy (via Gitea workflow):
#   NOMAD_ADDR=$NOMAD_ADDR nomad job run deploy/nomad/grassmann-distance.hcl
#
# Manual dispatch (for testing):
#   nomad job dispatch -meta job_id=<uuid> grassmann-distance

job "grassmann-distance" {
  datacenters = ["the-collective"]
  type        = "batch"

  parameterized {
    meta_required = ["job_id"]
  }

  # Constrain to GPU nodes with ROCm support
  constraint {
    attribute = "${meta.gpu}"
    value     = "true"
  }

  constraint {
    attribute = "${meta.rocm}"
    value     = "true"
  }

  group "worker" {
    count = 1

    stop_after_client_disconnect = "30s"

    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    network {
      mode = "host"
    }

    task "vector" {
      driver = "raw_exec"

      lifecycle {
        hook    = "poststart"
        sidecar = true
      }

      config {
        command = "/bin/sh"
        args    = ["-c", "mkdir -p ${NOMAD_ALLOC_DIR}/vector && exec /usr/local/bin/vector --config ${NOMAD_TASK_DIR}/vector.yaml"]
      }

      template {
        destination     = "local/vector.yaml"
        left_delimiter  = "[["
        right_delimiter = "]]"
        data            = <<EOF
data_dir: "${NOMAD_ALLOC_DIR}/vector"

sources:
  logs:
    type: "file"
    include: ["${NOMAD_ALLOC_DIR}/logs/*.std*.0"]
    read_from: "beginning"

transforms:
  enrich:
    type: "remap"
    inputs: ["logs"]
    source: |
      .job      = "grassmann-distance"
      .alloc_id = "${NOMAD_ALLOC_ID}"
      .cluster  = "the-collective"
      .job_id   = "${NOMAD_META_job_id}"

      path_parts = split!(string!(.file), "/")
      filename   = string!(path_parts[-1])
      name_parts = split!(filename, ".")
      .task      = name_parts[0]

      parsed, err = parse_json(.message)
      if err == null {
        .level = string(parsed.level) ?? "info"
      } else {
        .level = "info"
      }

sinks:
  loki:
    type: "loki"
    inputs: ["enrich"]
    endpoint: "[[ with secret "secret/data/nomad/forge" ]][[ .Data.data.LOKI_ENDPOINT ]][[ end ]]"
    tenant_id: "default"
    healthcheck:
      enabled: false
    encoding:
      codec: "text"
    labels:
      job:      "{{ job }}"
      task:     "{{ task }}"
      alloc_id: "{{ alloc_id }}"
      job_id:   "{{ job_id }}"
      cluster:  "the-collective"
      level:    "{{ level }}"
    compression: "gzip"
    batch:
      max_bytes: 1024000
      timeout_secs: 5
    buffer:
      type: "memory"
      max_events: 10000
EOF
      }

      vault {
        policies = ["forge"]
      }

      resources {
        cpu    = 50
        memory = 64
      }
    }

    task "worker" {
      driver = "raw_exec"

      # .sif lives on /nfs/images — mounted on the GPU node, no HTTP download
      config {
        command = "/usr/bin/singularity"
        args    = [
          "run",
          "--writable-tmpfs",
          "--bind", "/opt/rocm:/opt/rocm",
          "/nfs/images/grassmann-distance/worker.sif",
        ]
      }

      env {
        JOB_ID            = "${NOMAD_META_job_id}"
        WORKER_ID         = "nomad-${NOMAD_ALLOC_ID}"
        USE_GPU           = "true"
        JULIA_NUM_THREADS = "1"
        ROCM_PATH         = "/opt/rocm"
      }

      # Secrets from Vault — shared forge policy
      vault {
        policies = ["forge"]
      }

      template {
        data = <<EOT
{{ with secret "secret/data/nomad/forge" }}
MQTT_BROKER={{ .Data.data.MQTT_BROKER }}
MQTT_USER={{ .Data.data.MQTT_USER }}
MQTT_PASSWORD={{ .Data.data.MQTT_PASSWORD }}
{{ end }}
EOT
        destination = "secrets/env"
        env         = true
      }

      resources {
        cpu    = 2000
        memory = 4096
      }

      logs {
        max_files     = 3
        max_file_size = 10
      }

      kill_timeout = "30s"
      kill_signal  = "SIGTERM"
    }
  }
}
