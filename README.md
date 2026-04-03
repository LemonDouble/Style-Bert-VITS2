# Style-Bert-VITS2 (Fork)

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 기반으로 한 커스텀 TTS 패키지.

> 원본 Style-Bert-VITS2는 [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 참고하세요. 라이선스: AGPL-3.0

## 목표

1. **라이브러리화** — weight만 준비하면 `import`해서 바로 추론 가능한 패키지
2. **데이터셋 구축 파이프라인** — 영상+자막 → 학습 가능한 데이터셋
3. **API 서버** — 학습된 모델을 쉽게 서빙할 수 있는 추론 API
4. **한국어 확장** — 이중 모델 방식으로 한국어 TTS 지원 (추후)

## 아키텍처 결정

- **JP 모델**: Style-Bert-VITS2 JP-Extra (v2.7.0) 사용
  - 단일 BERT (DeBERTa JP), WavLM discriminator
  - 기존 pretrained 모델 그대로 활용
- **KO 모델**: 이중 모델 방식 — JP 모델과 별도로 한국어 전용 모델 운영 (예정)
  - JP 모델을 건드리지 않고 독립적으로 학습/최적화
  - 언어별 라우팅으로 전환

## 사용법 (목표 API)

```python
from style_bert_vits2 import TTS, Lang

# 공유 리소스(BERT, WavLM) 1회 로드
tts = TTS(device="cuda")

# Speaker는 자기 가중치(safetensors)만 로드
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

## 메모리 구조

```
TTS (공유 리소스 관리)
├── bert_jp: DeBERTa JP     ← 공유 (~1.3GB)
├── wavlm: WavLM            ← 공유 (~360MB)
│
├── speakers["elaina"]  → Generator 가중치만 (~240MB)
└── speakers["plana"]  → Generator 가중치만 (~240MB)
```

## 데이터셋 파이프라인

```
영상 + 자막 (.ass/.srt)
  → 1. 자막 타이밍 기반 음성 추출 (ffmpeg)
  → 2. UVR 배경음악 제거 (MDX23C-InstVoc HQ)
  → 3. 수동 QC (노이즈, 다른 캐릭터 대사 삭제)
  → 4. 리샘플링 (44.1kHz)
  → 5. 라벨 생성 (esd.list)
  → 6. 전처리 (BERT 피처 추출 등)
  → 학습 준비 완료
```

## 프로젝트 구조

```
Style-Bert-VITS2/
├── style_bert_vits2/           ← 핵심 패키지
│   ├── models/                 — 모델 정의 (models_jp_extra.py)
│   ├── nlp/                    — 언어별 NLP (G2P, BERT, 음소)
│   │   ├── japanese/
│   │   ├── english/
│   │   └── chinese/
│   └── constants.py
├── bert/                       ← BERT 모델 가중치 (gitignore)
│   └── deberta-v2-large-japanese-char-wwm/
├── slm/                        ← WavLM 가중치 (gitignore)
├── pretrained_jp_extra/        ← JP-Extra pretrained (gitignore)
├── model_assets/               ← fine-tuned 모델 (gitignore)
│   └── Elaina/
├── Data/                       ← 학습 데이터 (gitignore)
└── configs/
```

## 라이선스

- 코드: AGPL-3.0 (원본 Style-Bert-VITS2)
- 모델 가중치 (자체 학습): 별도 — 코드만 공개하면 AGPL 조건 충족
- 사전학습 모델 (BERT, WavLM 등): 각각 MIT

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
