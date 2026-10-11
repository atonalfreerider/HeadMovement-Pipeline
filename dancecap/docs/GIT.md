# Git policy for the workspace repository (dancecap pipeline)

This file covers the workspace repository (`HeadMovement`, remote `https://github.com/atonalfreerider/HeadMovement-Pipeline.git`).
The Unity viewer is a **separate** repository in the nested folder `head-movement/` (its own history, its own hooks, its
own `docs/GIT.md`); it is public and stricter. The workspace root ignores `head-movement/`.

> **Visibility (state of 2026-10-11).** The workspace remote is **private** (the owner changed the GitHub setting on
> 2026-10-11) and `git config hm.visibility` is `private` in the working clone. That setting is the owner's statement
> about a GitHub setting only the owner can see and change; agents never set it. The content rules below still hold:
> private does not make this a place for keys, gated body models, footage or likeness data, and parts of the repository
> (README, docs, code) may be shown to others, so keep it clean.

## What is tracked

The root `.gitignore` starts with `/*` and whitelists only: `.gitignore`, `.gitattributes`, `README.md`, `atlas_bridge/`,
`dancecap/`, `docs/images/`, `prototype/` and `tools/`. Run outputs (`dancecap/work/`, `atlas_bridge/sessions/*/work/`), weights, body
models (`models/`, `smplx/`), footage (`dances/`), third-party code (`external/`, `mamma*/`, `beat_this/`), virtual
environments and the nested `head-movement/` repository are all outside git on purpose.

Inside the tracked trees a second layer of ignore patterns keeps junk (profiler output, logs), audio/video, weights and
arrays, 3D captures, raster images and secrets from being added by accident. Ignore rules never affect files that are
already tracked.

## Branches

There is **one branch, `main`**, and `origin/main` is the same history. The history was cleaned by the owner (October 2026) and
force-pushed; the pre-cleanup history is kept only as the local tag `backup/main-2026-10-09`. Work lands on `main` directly, in
small logical commits. A short-lived `topic/<name>` branch is fine for a risky change; merge it with `git merge --ff-only`
and delete it.

Rules:

- **No force-push, no history rewrite** unless the owner asks for it. Agents never create, delete or move remote branches or tags.
- Checkpoints are annotated local tags `checkpoint-YYYY-MM-DD-<topic>` (for example `checkpoint-2026-10-08-pre-farm`); they stay local until the owner pushes.
- Keep the **dancer-specific take configs** (`dancecap/takes/*.toml`, which hold names, appearance edits, footage and
  track file names) in commits of their own where practical, so they can be dropped or moved without touching code.

## Commit conventions

- Subject: imperative, aim for 72 characters and stay under 110 (the hooks do not check this, and a few recent subjects are longer), area prefix when it helps (`progress:`, `cameras:`, `front_full:`,
  `remote:`, `pipeline:`, `monitor:`, `moves:`, `farm:`, `shape:`, `hair:`, `takes:`, `docs:`, `cli:`, `chore:`).
- A checkpoint snapshot says so with the prefix `checkpoint:`.
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
| Images, textures, portraits, hair grooms, point clouds, splats | Likeness of real people. The one exception is `docs/images/`: the README demo images of the Larissa and Kadu demo take, used with the artists' approval (see `allow.txt`). Images of any other take stay out. |
| Files above 5 MB, built executables (`PipelineMonitor.exe`), profiler output, run outputs under `work/` | Repository size and reproducibility. |
| The private forbidden-terms list | It lives at `~/.hm-forbidden-terms.txt`, outside every repository, and is never quoted anywhere. |

## Hooks (tools/githooks)

`pre-commit` and `commit-msg` call `tools/githooks/guard.py` (Python standard library only; the Unity repository carries an
identical copy: compare `python tools/githooks/guard.py --version` with the one in `head-movement/Tools/githooks/`).

**State of the working clone (2026-10-11).** `core.hooksPath` is `.git/hm-hooks`, a local directory (not tracked) whose
`pre-commit` and `commit-msg` are one-line wrappers that run the tracked hooks in `tools/githooks/`. **There is no pre-push
hook in that directory: the owner removed the push gate on purpose** (2026-10-11), so git does not check a push; the
commit hooks are the only automatic guard, which is one more reason never to bypass them. The file `tools/githooks/pre-push`
is still in the tree (it is the old gate: it refuses a push unless `hm.visibility` is `private` and audits every commit that
would be sent); it is not installed. `tools/install_hooks.sh` / `.ps1` point `core.hooksPath` at `tools/githooks/` and so
would bring it back: do not run the installer in this clone unless the owner wants the gate again.

- A fresh clone has no hooks until someone installs them: `sh tools/install_hooks.sh` (or `powershell -File tools/install_hooks.ps1`) sets
  `core.hooksPath` (including the push gate); `--check` shows the state, `--no-test` skips the self-test, `--uninstall` removes it (owner only).
  For a clone without the push gate, create a local hooks directory with the two wrappers instead, as the working clone does.
- Python 3.8+ is found as `python3`, `python`, then `.venv-nopo4d`; set `HM_GUARD_PYTHON` to force one.

The **workspace profile** blocks (rule ids in brackets): `key-file`, `secret` (patterns in added lines and the exact value of
any local `*-key.txt`), `gated-model`/`model-data`, `media`, `binary-asset` (a new image or 3D-model file), `likeness`
(portrait/texture/groom style data file names; source and docs are exempt), `size` (above 5 MB), `private-path` (the paths in the
table above).
It does **not** apply the private terms list line by line, because dancer-specific data is expected in this repository (set
`HM_GUARD_TERMS=1` to apply it that way anyway, for example to check a change before it is moved to the public Unity repository).
While `hm.visibility` is not `private` the guard also applies a **ratchet** (counted per file as lines added minus lines
removed): `local-path` blocks a commit that adds lines with an absolute path that contains a user directory, and `term` warns
about private terms added to files that `tools/githooks/terms-allow.txt` does not list (the take configs and the world manifests
are listed; `git config hm.guard.terms block` turns the warning into an error, `off` silences it). With `hm.visibility` set to
`private`, as it is now, the ratchet is off; keep absolute home paths and personal names out of new docs and code anyway.

- A line with the marker `hm-guard: ignore` is exempt from the secret-pattern and local-path rules (for test fixtures with fake keys).
- `tools/githooks/allow.txt` lists path globs exempt from `size` and `binary-asset` only; entries need the owner's approval,
  go in a commit of their own and apply from the next commit (the list is read from `HEAD`). Current entry: `docs/images/*`
  (the README demo images of the Larissa and Kadu demo take, used with the artists' approval; owner approval 2026-10-11).
- Audit without staging: `python tools/githooks/guard.py --profile workspace --worktree [paths]`; decoy tests: `... --self-test`
  (65 cases, both profiles, throw-away repository).
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
- A linked git worktree, if one exists, keeps its own uncommitted edits; ask the owner before committing or removing them. Agents work in the main checkout on `main`.

## Push policy

- Nothing is pushed without the owner's explicit go-ahead in the conversation. Agents prepare local commits only. There is no
  automatic push gate any more (see Hooks), so this rule is one of conduct, not a technical barrier.
- Before pushing, read `git log origin/main..main` and `git diff --stat origin/main..main`, and run the audit
  (`python tools/githooks/guard.py --profile workspace --worktree`) on anything new.
- Take configs, taxonomy data and world manifests name real people and are tracked on purpose, in this private repository. The
  public Unity repository must never receive them; its own guard rejects every private term.

## Line endings

`core.autocrlf=true` is set on the development machine, so Git may print "LF will be replaced by CRLF" warnings; they are
harmless. `.gitattributes` forces LF for `*.sh` and `tools/githooks/*` (shell scripts shipped to Linux pods and the hooks
themselves break with a CRLF shebang line).
