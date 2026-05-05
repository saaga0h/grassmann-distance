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

    task "worker" {
      driver = "raw_exec"

      # .sif lives on /nfs/images — mounted on the GPU node, no HTTP download
      config {
        command = "/usr/local/bin/singularity"
        args    = [
          "run",
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

        device "amd/gpu" {
          count = 1
        }
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
