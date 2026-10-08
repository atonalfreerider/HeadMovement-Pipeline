# Git policy for the workspace repository (dancecap pipeline)

This file covers the workspace repository (`HeadMovement`, remote `https://github.com/atonalfreerider/HeadMovement-Pipeline.git`).
The Unity viewer is a **separate** repository in the nested folder `head-movement/` (its own history, its own hooks, its
own `docs/GIT.md`); it is public and stricter. The workspace root ignores `head-movement/`.

> **Visibility warning.** On 2026-10-08 the workspace remote answered unauthenticated requests, i.e. it is **public**, although
> this repository is meant to be the private workspace. Everything already pushed is public. Until the owner has made the
> repository private (a GitHub setting only the owner can change) or has decided otherwise, treat everything committed here as
> potentially public. The hooks enforce that: `git config hm.visibility` is `public` (also when unset), the `pre-push` hook
> refuses every push, and `git config hm.visibility private` is how the owner records that the GitHub repository is private
> (see "Hooks"). Do not set it on your own; it is a statement about a GitHub setting only the owner can see and change.

## What is tracked

The root `.gitignore` starts with `/*` and whitelists only: `.gitignore`, `.gitattributes`, `README.md`, `atlas_bridge/`,
`dancecap/`, `prototype/` and `tools/`. Run outputs (`dancecap/work/`, `atlas_bridge/sessions/*/work/`), weights, body
models (`models/`, `smplx/`), footage (`dances/`), third-party code (`external/`, `mamma*/`, `beat_this/`), virtual
environments and the nested `head-movement/` repository are all outside git on purpose.

Inside the tracked trees a second layer of ignore patterns keeps junk (profiler output, logs), audio/video, weights and
arrays, 3D captures, raster images and secrets from being added by accident. Ignore rules never affect files that are
already tracked.

## Branches

| Branch | Role |
|---|---|
| `main` | The integration branch. Finished work lands here. |
| `phase2-wip` | The long-running work branch used so far. Land it on `main` as a fast-forward and keep it only as a pointer. |
| `claude/<name>` | Local scratch branches of agent worktrees under `.claude/worktrees/`. Never delete a worktree or reset its branch while it holds uncommitted work. |
| `topic/<short-name>` | Optional short-lived branch for a risky or long change. |

Rules:

- Land work with `git switch main && git merge --ff-only <branch>`. `git branch -f main <branch>` is acceptable only when
  `git merge-base --is-ancestor main <branch>` is true and `main` is not checked out in any worktree.
- **No force-push, no history rewrite** of anything that was pushed. Agents never create, delete or move remote branches or tags.
- Checkpoints are annotated local tags `checkpoint-YYYY-MM-DD-<topic>` (for example `checkpoint-2026-10-08-pre-farm`); they stay local until the owner pushes.
- Keep the **dancer-specific take configs** (`dancecap/takes/*.toml`, which hold names, appearance edits, footage and
  track file names) in a commit of their own, last in a series, so they can be dropped or moved without touching code.

## Commit conventions

- Subject: imperative, aim for 72 characters and never exceed 110, area prefix when it helps (`progress:`, `cameras:`, `front_full:`,
  `remote:`, `pipeline:`, `monitor:`, `moves:`, `farm:`, `shape:`, `hair:`, `takes:`, `docs:`, `cli:`, `chore:`). The 2026-10-08
  series was written before this was checked; the 2026-10-09 history cleanup shortened its longer subjects to 110 characters (the original subject is kept in the commit body).
- A checkpoint snapshot says so with the prefix `checkpoint:` (new commits) or the suffix `(checkpoint)` (the 2026-10-08 series).
- Body: what changed and why; quote the user's request (short, in quotation marks) when it motivated the change; list what was left out.
- One logical change per commit. Mixed-concern files (for example `cli.py`, `CONTRACTS.md`) are committed after the modules they wire or document.
- Agent commits end with the trailer line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Commit messages are scanned for secrets by the commit-msg hook.

## What must never be committed, and why

| Never | Why |
|---|---|
| API keys, tokens, passwords, `.env`, `*-key.txt`, certificates | Security. Code reads key **files** from the Desktop at run time; it never contains a value. |
| Gated body models (SMPL-X, FLAME, MANO) and files derived from them; weights, `.npz`/`.pkl`/checkpoints | The licences forbid redistribution; size. They are copied to pods from the local machine, never through git. |
| Footage, audio, narration, renders (`.mp4`, `.wav`, ...) | Copyright and personal data. |
| Anything under `dancecap/work/`, `atlas_bridge/sessions/*/work/`, `models/`, `weights/`, `smplx/`, `dances/`, `footage/` (the hook refuses it even with `git add -f`) | Run outputs, gated models and footage stay outside git. |
| Images, textures, portraits, hair grooms, point clouds, splats | Likeness of real people. |
| Files above 5 MB, built executables (`PipelineMonitor.exe`), profiler output, run outputs under `work/` | Repository size and reproducibility. |
| The private forbidden-terms list | It lives at `~/.hm-forbidden-terms.txt`, outside every repository, and is never quoted anywhere. |

## Hooks (tools/githooks)

`pre-commit` and `commit-msg` call `tools/githooks/guard.py` (Python standard library only; the Unity repository carries an
identical copy: compare `python tools/githooks/guard.py --version` with the one in `head-movement/Tools/githooks/`).

- Install once per clone, and again in linked worktrees that predate the hooks (the script handles those): `sh tools/install_hooks.sh`
  or `powershell -File tools/install_hooks.ps1`. It only sets `core.hooksPath`; `--check` shows the state, `--no-test` skips the self-test, `--uninstall` removes it (owner only).
- Python 3.8+ is found as `python3`, `python`, then `.venv-nopo4d`; set `HM_GUARD_PYTHON` to force one.

The **workspace profile** blocks (rule ids in brackets): `key-file`, `secret` (patterns in added lines and the exact value of
any local `*-key.txt`), `gated-model`/`model-data`, `media`, `binary-asset` (a new image or 3D-model file), `likeness`
(portrait/texture/groom style data file names; source and docs are exempt), `size` (above 5 MB).
It does **not** apply the private terms list line by line, because dancer-specific data is expected in this repository (set
`HM_GUARD_TERMS=1` to apply it that way anyway, for example to check a commit before it is moved to the public repository).
Because the remote is **public** (`hm.visibility` is not `private`) it applies a **ratchet** instead, counted per file as lines
added minus lines removed, so editing a line that already carries the term or path is free:

- `local-path` (blocks): a commit may not add lines with an absolute path that contains a user directory.
- `term` (warns): a commit may not add lines with private terms to a file that `tools/githooks/terms-allow.txt` does not list
  (the take configs and the world manifests are listed). The warning does not stop the commit, because the take names run through
  the code base; `git config hm.guard.terms block` (or `HM_GUARD_TERMS_MODE=block`) turns it into an error, `off` silences it.
  The private terms list is `~/.hm-forbidden-terms.txt`, as in the Unity repository.
- `private-path` (blocks): the paths in the table above.
- `push` (the `pre-push` hook): **no push at all** while `hm.visibility` is not `private`; deletions and non-fast-forward
  updates are refused in any case; once the repository is recorded as private, every commit that would be sent is audited again
  (secrets, media, models, private paths), which also catches commits made with `--no-verify`.

- A line with the marker `hm-guard: ignore` is exempt from the secret-pattern and local-path rules (for test fixtures with fake keys).
- `tools/githooks/allow.txt` lists path globs exempt from `size` and `binary-asset` only; entries need the owner's approval,
  go in a commit of their own and apply from the next commit.
- Audit without staging: `python tools/githooks/guard.py --profile workspace --worktree [paths]`; decoy tests: `... --self-test`
  (65 cases, both profiles, throw-away repository). Check what a push would send without pushing, against the real repository:
  `printf 'refs/heads/main %s refs/heads/main %s\n' $(git rev-parse main) $(git rev-parse origin/main) | python tools/githooks/guard.py --profile workspace --pre-push origin x`
  (set `HM_GUARD_VISIBILITY=private` to see the content audit that would apply after the repository is private).
- Blocked? Unstage the path (`git restore --staged <path>`), fix or remove the content, or git-ignore it. Do not work around the hook.
- **Emergency bypass, repository owner only:** `HM_GUARD_BYPASS="reason" git commit ...` or `git commit --no-verify`. The
  first form warns and logs the reason to `.git/hm-guard-bypass.log`. Agents never bypass the hooks.

## How workflows commit

- A **git agent** commits logical groups at the end of each workflow stage. Workflows do not push and do not commit each
  other's files. Hooks apply to agents and owner alike; agents never use `--no-verify` or `HM_GUARD_BYPASS`.
- Stage with **explicit path lists** (`git add <paths>`), never `-A`, `.` or `-f`; read `git diff --cached --stat` first.
- Python files must parse before they are committed (`python -m py_compile`), and the tests of the touched module should pass.
- A file modified in the last few minutes, or owned by a running workflow (the farm, the full-dance fit, face shape, film),
  is **left for the next checkpoint** and listed as such. A checkpoint snapshot is labelled `checkpoint:` in the subject.
- Dependency order for a series: ignore rules, then low-level modules (progress, cameras), then the stages that import them,
  then the pipeline, the monitor and the launcher, and last the docs and the CLI that wire everything together.
- A linked worktree under `.claude/worktrees/` keeps its own uncommitted edits; ask the owner before committing or removing them.

## Push policy

- Nothing is pushed without the owner's explicit go-ahead in the conversation. Agents prepare local commits only.
- Because the remote is currently public (see the warning above), the owner should decide first whether to make it private
  or to move dancer-specific files (take configs, taxonomy data) out before pushing the new work. The `pre-push` hook
  enforces this: it refuses while `hm.visibility` is not `private`. The take configs sit in their own commit (`takes:`, last in
  the series) so they can be dropped without touching code; the tests that read the real take files skip when those files
  are absent, so a checkout without that commit still runs the rest of the suite.
- Before pushing: run the audit, inspect `git log origin/main..main` and `git diff --stat origin/main..main`.
- Older commits contain dancer-specific material that predates these rules. Cleaning that needs a history rewrite and a
  force-push, which is the owner's decision.

## Line endings

`core.autocrlf=true` is set on the development machine, so Git may print "LF will be replaced by CRLF" warnings; they are
harmless. `.gitattributes` forces LF for `*.sh` and `tools/githooks/*` (shell scripts shipped to Linux pods and the hooks
themselves break with a CRLF shebang line).
