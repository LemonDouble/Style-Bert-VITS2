# Style-Bert-VITS2 (Fork)

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 기반으로 한 커스텀 TTS 패키지.

> 원본 Style-Bert-VITS2는 [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 참고하세요. 라이선스: AGPL-3.0

## 목표

1. **라이브러리화** — `pip install` + fine-tuned 모델만 준비하면 바로 추론 가능
2. **API 서버** — 학습된 모델을 쉽게 서빙
3. **한국어 확장** — 이중 모델 방식으로 한국어 TTS 지원 (추후)

## 아키텍처 결정

- **JP 모델**: Style-Bert-VITS2 JP-Extra (v2.7.0) 사용
  - 단일 BERT (DeBERTa JP), WavLM discriminator
  - 기존 pretrained 모델 그대로 활용
- **KO 모델**: 이중 모델 방식 — JP 모델과 별도로 한국어 전용 모델 운영 (예정)
  - JP 모델을 건드리지 않고 독립적으로 학습/최적화
  - 언어별 라우팅으로 전환

## 추론

### 설치

```bash
pip install style-bert-vits2
```

BERT, WavLM, pretrained 등 공유 리소스는 패키지가 자동으로 다운로드합니다.
유저는 **fine-tuned 모델(safetensors)만** 준비하면 됩니다.

### 기본 사용법

```python
from style_bert_vits2 import TTS, Lang

# 공유 리소스 자동 다운로드 + 메모리 로드
tts = TTS(device="cuda")

# Speaker는 fine-tuned 가중치(safetensors)만 로드
elaina = tts.load("elaina", "./models/Elaina")

# 기본 사용
audio = elaina.generate("先生、大丈夫ですか？", lang=Lang.JA, style=elaina.styles.Neutral)
audio.save("output.wav")

# 파라미터 조절
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

```python
# prepare() — 가중치 다운로드만 (GPU 불필요, CI/빌드 단계용)
TTS.prepare()

# TTS() — 다운로드 + 메모리 로드 (이미 다운로드 되어있으면 로드만)
tts = TTS(device="cuda")
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
└── speakers["plana"]  → fine-tuned 가중치만 (~240MB, 유저 제공)
```

## 라이선스

- 코드: AGPL-3.0 (원본 Style-Bert-VITS2)
- 모델 가중치 (자체 학습): 별도 — 코드만 공개하면 AGPL 조건 충족
- 사전학습 모델 (BERT, WavLM 등): 각각 MIT

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
