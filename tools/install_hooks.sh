#!/bin/sh
# Install the hm-guard git hooks (tools/githooks/) into THIS clone.  Local git config only: nothing is
# pushed or shared, so run it once per clone (and once per linked worktree that predates the hooks).
#
#   tools/install_hooks.sh               install, then run the guard self-test
#   tools/install_hooks.sh --no-test     install without the self-test
#   tools/install_hooks.sh --check       show the current state and exit
#   tools/install_hooks.sh --uninstall   remove the setting (repository owner only)
#
# Rules, the owner-only bypass and the commit conventions: dancecap/docs/GIT.md
set -e
root=$(git rev-parse --show-toplevel)
cd "$root"
hooks=tools/githooks
mode=install
for a in "$@"; do
    case "$a" in
        --check) mode=check ;;
        --no-test) mode=notest ;;
        --uninstall) mode=uninstall ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

. "$root/$hooks/hm-guard.sh"

state() {
    cur=$(git config --get core.hooksPath || true)
    echo "core.hooksPath = ${cur:-<unset>}"
    for h in pre-commit commit-msg; do
        if [ -f "$hooks/$h" ]; then echo "  $hooks/$h present"; else echo "  $hooks/$h MISSING"; fi
    done
    if py=$(hm_guard_python); then
        echo "python: $py ($("$py" "$hooks/guard.py" --version))"
    else
        echo "python: NOT FOUND (set HM_GUARD_PYTHON)"
    fi
}

if [ "$mode" = check ]; then state; exit 0; fi

if [ "$mode" = uninstall ]; then
    git config --unset core.hooksPath || true
    echo "hm-guard hooks uninstalled from this clone."
    state
    exit 0
fi

chmod +x "$hooks/pre-commit" "$hooks/commit-msg" "$hooks/guard.py" 2>/dev/null || true
git config core.hooksPath "$hooks"

# Remember a working interpreter for git clients whose PATH has no Python (PowerShell, IDEs, services).
exe=""
if py=$(hm_guard_python); then
    exe=$("$py" -c "import sys; print(sys.executable)" | tr -d '\r')
    exe=$(cygpath -m "$exe" 2>/dev/null || printf '%s' "$exe")
    printf '%s\n' "$exe" > "$(git rev-parse --absolute-git-dir)/hm-guard-python"
fi

# linked worktrees whose checked-out commit does not contain tools/githooks: point them at this clone's copy
if [ "$(git config --get extensions.worktreeconfig || true)" = true ]; then
    git worktree list --porcelain | sed -n 's/^worktree //p' | while read -r wt; do
        [ "$wt" = "$root" ] && continue
        if [ ! -d "$wt/$hooks" ]; then
            git -C "$wt" config --worktree core.hooksPath "$root/$hooks"
            [ -n "$exe" ] && printf '%s\n' "$exe" > "$(git -C "$wt" rev-parse --absolute-git-dir)/hm-guard-python"
            echo "linked worktree $wt -> $root/$hooks"
        fi
    done
fi

echo "hm-guard hooks installed (core.hooksPath = $hooks)."
state

if [ "$mode" != notest ]; then
    py=$(hm_guard_python) || { echo "no python: skipping the self-test" >&2; exit 1; }
    "$py" "$hooks/guard.py" --self-test
fi
