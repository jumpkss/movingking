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

`.claude/vendor/im-not-ai/`에 im-not-ai(한글 AI 티 제거기) 플러그인을 두고,
스킬(`humanize-korean`, `humanize`, `humanize-scan`, `humanize-redo`)과
서브에이전트를 심링크로 연결했습니다. 자세한 내용은 `.claude/README.md`를 보세요.
