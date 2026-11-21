# scripts/studio/env.sh: sourced (not executed) by every script in this directory. Loads scripts/studio/local.env
# (copy local.env.example and fill it in; the file is gitignored and sync.sh carries it to the inference Mac), then
# exports the machine layout the scripts use:
#   STUDIO_HOST   ssh host of the inference Mac            INK2Q_WORK  fast working directory on that Mac
#   STUDIO_REPO   this repo's checkout on that Mac         INK2Q_ROOT  bulk data root: source, models, reports
#   STUDIO_HOME   studio-inference workspace (default $HOME/studio-inference)
#   UV            uv binary (default: uv on PATH)          TF_FORK, TF_PACK  derived from INK2Q_WORK
# Optional, checked by the scripts that use them: OMLX_BIN, STUDIO_ALLOWED_HOSTS (see local.env.example).

_ink2q_env_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$_ink2q_env_dir/local.env" ] && . "$_ink2q_env_dir/local.env"
unset _ink2q_env_dir

_ink2q_hint="is not set; copy scripts/studio/local.env.example to scripts/studio/local.env and fill it in"
: "${STUDIO_HOST:?$_ink2q_hint}" "${STUDIO_REPO:?$_ink2q_hint}" "${INK2Q_WORK:?$_ink2q_hint}" "${INK2Q_ROOT:?$_ink2q_hint}"

# need VAR...: exit, naming the variable, when any is unset or empty (for the optional settings a script uses).
need() {
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "$v $_ink2q_hint" >&2; exit 1; }
  done
}

STUDIO_HOME=${STUDIO_HOME:-$HOME/studio-inference}
UV=${UV:-$(command -v uv || true)}
TF_FORK=$INK2Q_WORK/tf-fork
TF_PACK=$INK2Q_WORK/tf-pack
export STUDIO_HOST STUDIO_REPO INK2Q_WORK INK2Q_ROOT STUDIO_HOME UV TF_FORK TF_PACK
