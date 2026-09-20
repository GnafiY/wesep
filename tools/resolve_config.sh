#!/bin/bash

# Resolve runtime script variables from command-line values, YAML config, and
# recipe fallbacks. This file is meant to be sourced by recipe run.sh scripts.

resolve_yaml_value() {
  local current_value=$1
  local config=$2
  local key=$3
  local fallback=$4

  if [ -n "${current_value}" ]; then
    echo "${current_value}"
    return
  fi

  python -c 'import json, sys, yaml
config, key, fallback = sys.argv[1:4]
with open(config, encoding="utf-8") as stream:
    data = yaml.safe_load(stream) or {}
value = data.get(key, fallback)
if isinstance(value, (list, dict)):
    print(json.dumps(value, separators=(",", ":")))
elif isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
' "${config}" "${key}" "${fallback}"
}

resolve_fs() {
  local current_value=$1
  local config=$2
  local fallback=$3

  if [ -n "${current_value}" ]; then
    echo "${current_value}"
    return
  fi

  python -c 'import sys, yaml
config, fallback = sys.argv[1:3]
with open(config, encoding="utf-8") as stream:
    data = yaml.safe_load(stream) or {}
sample_rate = (data.get("dataset_args") or {}).get("resample_rate")
if sample_rate is None:
    print(fallback)
else:
    sample_rate = int(sample_rate)
    if sample_rate == 16000:
        print("16k")
    elif sample_rate == 8000:
        print("8k")
    else:
        raise SystemExit(f"Unsupported sample rate for fs: {sample_rate}")
' "${config}" "${fallback}"
}

true
