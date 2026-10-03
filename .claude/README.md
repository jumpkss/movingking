# .claude 디렉터리 안내

이 폴더는 Claude Code가 이 저장소에서 자동으로 읽는 설정입니다.
**설치 절차가 필요 없습니다.** 웹/앱에서 이 저장소로 세션을 열면 그대로 적용됩니다.

## 들어 있는 것

| 경로 | 내용 |
|---|---|
| `skills/` | Superpowers 스킬 14종 (Claude가 읽는 작업 매뉴얼) |
| `SUPERPOWERS-LICENSE.txt` | 원저작자 MIT 라이선스 전문 |
| `vendor/im-not-ai/` | humanize-korean 플러그인 원본 (스킬·에이전트·스크립트·라이선스) |
| `skills/humanize*` | 위 원본의 스킬 4종을 가리키는 심링크 |
| `agents/*.md` | 위 원본의 서브에이전트 4종을 가리키는 심링크 |

## 출처

- 프로젝트: Superpowers — https://github.com/obra/superpowers
- 버전: 6.3.0
- 저작자: Jesse Vincent
- 라이선스: MIT (`SUPERPOWERS-LICENSE.txt` 참조)

`skills/` 아래 파일은 위 저장소에서 그대로 복사한 것이며 수정하지 않았습니다.

## humanize-korean (한글 AI 티 제거기)

- 프로젝트: im-not-ai — https://github.com/epoko77-ai/im-not-ai
- 버전: 2.3.2 (커밋 `2f3d943`)
- 저작자: epoko77-ai
- 라이선스: MIT (`vendor/im-not-ai/LICENSE` 참조)
- 사용법: `/humanize-korean` (전수 윤문), `/humanize-scan` (AI 티 점수만 확인),
  또는 "이 글 AI 티 없애줘"처럼 자연어로 요청

원본에서 `.claude-plugin/`, `skills/`, `agents/`, `scripts/`, `LICENSE`만 그대로
복사했고 수정하지 않았습니다.

Superpowers처럼 `skills/`에 바로 복사하지 않고 `vendor/`에 두고 심링크로 연결한
이유: 이 스킬은 `.claude-plugin/` 폴더가 나올 때까지 경로를 거슬러 올라가
`scripts/`(정량 채점·검증 게이트)를 찾습니다. 스킬 폴더만 복사하면 이 탐색이
실패해 게이트가 동작하지 않습니다. 원본 `install.sh`도 같은 이유로 심링크를
씁니다. 서브에이전트는 `install.sh` 기본값과 같은 4종만 연결했습니다.

윤문 작업 폴더 `_workspace/`는 `.gitignore`에 추가했습니다.

갱신하려면:

```bash
git clone --depth 1 https://github.com/epoko77-ai/im-not-ai /tmp/im-not-ai
rm -rf .claude/vendor/im-not-ai && mkdir -p .claude/vendor/im-not-ai
cp -r /tmp/im-not-ai/{.claude-plugin,skills,agents,scripts,LICENSE} .claude/vendor/im-not-ai/
```

새 버전에서 스킬·에이전트 이름이 바뀌었으면 심링크도 맞춰 주세요. 제거하려면
`.claude/vendor/im-not-ai/`, `.claude/skills/humanize*`, `.claude/agents/`를 지우면 됩니다.

> 주의: Superpowers 갱신 절차의 `rm -rf .claude/skills`는 humanize 심링크까지
> 지웁니다. 갱신 뒤 심링크를 다시 만들어 주세요.

## 왜 플러그인 설치가 아니라 복사인가

Claude Code 문서(v2.1.195 이후) 기준으로, 프로젝트 `.claude/settings.json`에
`enabledPlugins`만 적어두어도 **GitHub 같은 외부 출처의 플러그인은 자동 설치되지
않습니다.** 각자 `claude plugin install`을 직접 실행해야 합니다.

실제로 이 방식을 이 저장소에서 시험해 본 결과 설치되지 않는 것을 확인했습니다.
반면 `.claude/skills/`에 직접 둔 스킬은 설치 절차 없이 바로 로드됩니다.
그래서 복사 방식을 택했습니다.

대신 플러그인이 제공하던 SessionStart 훅(세션 시작 시 `using-superpowers`를
읽게 하는 기능)은 빠져 있습니다. 같은 효과를 저장소 루트 `CLAUDE.md`의 안내
문구로 대체했습니다.

## 최신 버전으로 갱신하려면

Claude에게 "superpowers 스킬 최신으로 갱신해줘"라고 요청하거나, 직접 한다면:

```bash
git clone --depth 1 https://github.com/obra/superpowers /tmp/superpowers
rm -rf .claude/skills
mkdir -p .claude/skills
cp -r /tmp/superpowers/skills/. .claude/skills/
```

갱신 후에는 `.claude/README.md`의 버전 표기도 함께 고쳐 주세요.

## 사용을 중단하려면

`.claude/skills/` 폴더를 지우고, `CLAUDE.md`의 Skills 절을 삭제하면 됩니다.
