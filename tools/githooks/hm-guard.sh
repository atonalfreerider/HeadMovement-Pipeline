# shellcheck shell=sh
# Shared shell helpers for the hm-guard hooks.  Sourced by pre-commit and commit-msg, never run directly.
# Expects: here (this directory), HM_GUARD_PROFILE (workspace | unity).

# Candidate interpreters, most specific first, one per line.  Git may be started from PowerShell, an IDE or a
# service whose PATH has no Python, so besides PATH this also tries the interpreter recorded by install_hooks.sh,
# the project virtual environment and the usual per-user install locations.
hm_guard_candidates() {
    root=$(pwd)
    saved=""
    gd=${GIT_DIR:-.git}
    if [ -f "$gd/hm-guard-python" ]; then
        saved=$(sed -n '1p' "$gd/hm-guard-python" | tr -d '\r')
    fi
    for c in "$HM_GUARD_PYTHON" "$saved" python3 python \
        "$root/.venv-nopo4d/Scripts/python.exe" "$root/../.venv-nopo4d/Scripts/python.exe" \
        "$HOME/miniconda3/python.exe" "$HOME/anaconda3/python.exe" "$USERPROFILE/miniconda3/python.exe" \
        "$LOCALAPPDATA/Programs/Python/Python312/python.exe" "$LOCALAPPDATA/Programs/Python/Python311/python.exe" \
        "$LOCALAPPDATA/Programs/Python/Python310/python.exe" /usr/bin/python3 /usr/local/bin/python3; do
        if [ -n "$c" ]; then
            printf '%s\n' "$c"
        fi
    done
}

# Print the first working Python 3.8+ (used by install_hooks.sh; the hooks launch guard.py directly to save a process start).
hm_guard_python() {
    cands=$(hm_guard_candidates)
    while IFS= read -r c; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1; then
            printf '%s\n' "$c"
            return 0
        fi
    done <<EOF
$cands
EOF
    return 1
}

# hm_guard <hook name> <guard.py arguments...>
hm_guard() {
    hook=$1
    shift
    # Owner-only emergency bypass (documented in GIT.md).  Visible and logged on purpose.
    if [ -n "$HM_GUARD_BYPASS" ]; then
        gd=$(git rev-parse --git-dir 2>/dev/null)
        if [ -n "$gd" ]; then
            printf '%s %s BYPASS: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$hook" "$HM_GUARD_BYPASS" >> "$gd/hm-guard-bypass.log"
        fi
        echo "hm-guard: $hook check BYPASSED ($HM_GUARD_BYPASS); logged in the git dir" >&2
        return 0
    fi
    cands=$(hm_guard_candidates)
    while IFS= read -r c; do
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" "$here/guard.py" --profile "$HM_GUARD_PROFILE" "$@"
        rc=$?
        # 127 / 9009: the interpreter could not run (missing, Windows Store stub): try the next candidate
        if [ "$rc" -ne 127 ] && [ "$rc" -ne 9009 ]; then
            return "$rc"
        fi
    done <<EOF
$cands
EOF
    if command -v py >/dev/null 2>&1; then
        py -3 "$here/guard.py" --profile "$HM_GUARD_PROFILE" "$@"
        rc=$?
        if [ "$rc" -ne 127 ] && [ "$rc" -ne 9009 ]; then
            return "$rc"
        fi
    fi
    echo "hm-guard: no working Python 3.8+ found, so the $hook check cannot run and the commit is refused." >&2
    echo "          Put python on PATH or set HM_GUARD_PYTHON=<path to python>.  See GIT.md (docs/GIT.md or dancecap/docs/GIT.md)." >&2
    return 1
}
