WESEP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export WESEP_ROOT
export PATH=${WESEP_ROOT}:$PATH

# Use UTF-8 in Python when recipes run under LC_ALL=C.
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${WESEP_ROOT}:${PYTHONPATH:-}
