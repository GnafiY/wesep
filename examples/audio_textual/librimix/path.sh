WESEP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export WESEP_ROOT
export PATH=${WESEP_ROOT}:$PATH
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${WESEP_ROOT}:${PYTHONPATH:-}
