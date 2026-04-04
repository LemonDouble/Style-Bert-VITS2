# Style-Bert-VITS2 (Fork)

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 기반으로 한 일본어 TTS 라이브러리.

> 원본 Style-Bert-VITS2는 [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 참고하세요. 라이선스: AGPL-3.0

## 특징

- **라이브러리화** — `pip install` + fine-tuned 모델만 준비하면 바로 추론 가능
- **JP-Extra 모델** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP + WavLM
- **영어→카타카나 자동 변환** — 22만 엔트리 외래어 사전 룩업 (의존성 없음)

## 설치

```bash
pip install style-bert-vits2
```

BERT, WavLM 등 공유 리소스는 자동 다운로드됩니다. **fine-tuned 모델(safetensors)만** 준비하세요.

## 사용법

```python
from style_bert_vits2 import TTS, Lang

tts = TTS(device="cuda")
elaina = tts.load("elaina", "./models/Elaina")

audio = elaina.generate("先生、大丈夫ですか？", lang=Lang.JA, style=elaina.styles.Neutral)
audio.save("output.wav")
```

파라미터 조절:

```python
audio = elaina.generate(
    "先生、今日も頑張りましょう！",
    lang=Lang.JA,
    style=elaina.styles.Neutral,
    speed=0.9,
    sdp_ratio=0.2,
    noise=0.6,
    noise_w=0.8,
    pitch_scale=1.0,
    intonation_scale=1.0,
    style_weight=1.0,
)
```

### Docker / 서버 환경

```dockerfile
# 빌드 시점 — 이미지에 가중치 포함 (GPU 불필요)
RUN python -c "from style_bert_vits2 import TTS; TTS.prepare()"

# 실행 시점 — 다운로드 없이 바로 서빙
CMD ["python", "server.py"]
```

| 메서드 | 다운로드 | 메모리 로드 | GPU 필요 | 용도 |
|--------|----------|-------------|----------|------|
| `TTS.prepare()` | O | X | X | Docker 빌드, CI |
| `TTS(device=...)` | O (없으면) | O | O | 추론 |

### 메모리 구조

```
TTS (공유 리소스 — 자동 관리)
├── bert_jp: DeBERTa JP     ← 자동 다운로드 (~1.3GB)
├── wavlm: WavLM            ← 자동 다운로드 (~360MB)
│
├── speakers["elaina"]  → fine-tuned 가중치만 (~240MB, 유저 제공)
└── speakers["plana"]   → fine-tuned 가중치만 (~240MB, 유저 제공)
```

## 영어→카타카나 변환

일본어 텍스트에 포함된 영어 단어를 자동으로 카타카나 외래어로 변환합니다.

- `hello` → `ハロー`, `computer` → `コンピュータ`, `meeting` → `ミーティング`
- 사전에 없는 단어는 기존 동작 유지 (pyopenjtalk 처리)
- 데이터 출처: [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) (GPL-3.0)

## 라이선스

- 코드: AGPL-3.0 (원본 Style-Bert-VITS2)
- 모델 가중치 (자체 학습): 별도 — 코드만 공개하면 AGPL 조건 충족
- 사전학습 모델 (BERT, WavLM 등): 각각 MIT
- 영어→카타카나 사전 데이터: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))
  - 원본 데이터 소스: JMdict (CC BY-SA 4.0), CMUdict (BSD), Wikipedia/Wiktionary (CC BY-SA 3.0), Britfone, JTCA, LREC'14

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — 영어→카타카나 외래어 사전 데이터
