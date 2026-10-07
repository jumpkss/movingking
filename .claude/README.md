# .claude 디렉터리 안내

이 폴더는 Claude Code가 이 저장소에서 자동으로 읽는 설정입니다.
**설치 절차가 필요 없습니다.** 웹/앱에서 이 저장소로 세션을 열면 그대로 적용됩니다.

## 들어 있는 것

| 경로 | 내용 |
|---|---|
| `skills/` | Superpowers 스킬 14종 (Claude가 읽는 작업 매뉴얼) |
| `SUPERPOWERS-LICENSE.txt` | 원저작자 MIT 라이선스 전문 |
| `vendor/im-not-ai/` | humanize-korean 원본 (스킬·스크립트·에이전트·라이선스) |
| `skills/humanize*` (4개) | 위 원본의 스킬을 가리키는 심링크 |
| `agents/` | humanize-korean 런타임 에이전트 3종 (monolith·diagnostician·finalizer) |

## 출처

- 프로젝트: Superpowers — https://github.com/obra/superpowers
- 버전: 6.3.0
- 저작자: Jesse Vincent
- 라이선스: MIT (`SUPERPOWERS-LICENSE.txt` 참조)

`skills/` 아래 파일은 위 저장소에서 그대로 복사한 것이며 수정하지 않았습니다.

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

## Superpowers를 최신 버전으로 갱신하려면

Claude에게 "superpowers 스킬 최신으로 갱신해줘"라고 요청하거나, 직접 한다면:

```bash
git clone --depth 1 https://github.com/obra/superpowers /tmp/superpowers
find .claude/skills -mindepth 1 -maxdepth 1 ! -name 'humanize*' -exec rm -rf {} +
cp -r /tmp/superpowers/skills/. .claude/skills/
```

갱신 후에는 `.claude/README.md`의 버전 표기도 함께 고쳐 주세요.

## humanize-korean (AI 티 나는 한글 글 윤문)

- 프로젝트: im-not-ai — https://github.com/epoko77-ai/im-not-ai
- 버전: 2.3.2 (커밋 2f3d943)
- 저작자: epoko77-ai
- 라이선스: MIT (`vendor/im-not-ai/LICENSE` 참조)

플러그인 `humanize-korean@im-not-ai`의 설치본을 수정 없이 복사했습니다.

### 쓰는 법

- `/humanize <텍스트 또는 파일 경로>`: 전체 윤문
- `/humanize-scan`: AI 티가 얼마나 있는지만 빠르게 점검
- `/humanize-redo`: 직전 윤문 결과를 한 번 더 다듬기
- "AI 티 없애줘"처럼 말로 요청해도 `humanize-korean` 스킬이 동작합니다.

윤문 중간 산출물은 작업 폴더의 `_workspace/`에 생기며 `.gitignore`로 제외했습니다.

### Superpowers와 다른 점: 왜 심링크인가

이 스킬은 실행할 때 자기 폴더에서 위로 올라가며 `.claude-plugin/` 폴더를 찾고,
그 옆의 `scripts/`(정량 점수·변경률 검사 파이썬 스크립트)를 씁니다. 스킬 폴더만
`skills/`에 복사하면 이 스크립트를 찾지 못합니다. 그래서 플러그인 구조 그대로
`vendor/im-not-ai/`에 두고 `skills/`에는 심링크만 걸었습니다. 스킬 파일은
고치지 않았습니다.

에이전트는 스킬이 실행 중에 부르는 3종만 `agents/`에 복사했습니다. 나머지 6종은
분류 체계 개발용이라 `vendor/im-not-ai/agents/`에만 있습니다.

### 갱신하려면

```bash
git clone --depth 1 https://github.com/epoko77-ai/im-not-ai /tmp/im-not-ai
rm -rf .claude/vendor/im-not-ai && mkdir -p .claude/vendor/im-not-ai
cp -r /tmp/im-not-ai/{.claude-plugin,skills,scripts,agents,LICENSE} .claude/vendor/im-not-ai/
for a in humanize-monolith humanize-diagnostician humanize-finalizer; do
  cp .claude/vendor/im-not-ai/agents/$a.md .claude/agents/
done
```

갱신 후에는 위 버전 표기도 함께 고쳐 주세요.

### 중단하려면

`.claude/vendor/im-not-ai/`, `.claude/skills/humanize*` 심링크 4개,
`.claude/agents/humanize-*.md` 3개를 지우면 됩니다.

## Superpowers 사용을 중단하려면

`.claude/skills/` 폴더를 지우고, `CLAUDE.md`의 Skills 절을 삭제하면 됩니다.
