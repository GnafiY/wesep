#!/usr/bin/env bash

WESEP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export WESEP_ROOT
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="${WESEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${WESEP_ROOT}:${PATH}"
