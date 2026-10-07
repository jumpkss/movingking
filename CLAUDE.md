# movingking

## Skills

This repository vendors the Superpowers skill library into `.claude/skills/`.
See `.claude/README.md` for provenance, update, and removal instructions.

At the start of a session, consult the `using-superpowers` skill before
responding. It establishes when and how to invoke the other skills.

Skill names here are **not namespaced**. Upstream text refers to skills as
`superpowers:brainstorming`; in this repository the same skill is invoked as
`brainstorming`. Drop the `superpowers:` prefix when following cross-references
inside the skill files.

### humanize-korean

`humanize-korean` (im-not-ai) is vendored under `.claude/vendor/im-not-ai/` and
exposed through symlinks in `.claude/skills/` plus three runtime agents in
`.claude/agents/`. Use it when the user asks to remove AI-sounding style from
Korean text (`/humanize`, `/humanize-scan`, `/humanize-redo`). See
`.claude/README.md` for details.
