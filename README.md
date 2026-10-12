# Normal-only Defect Screening (MVTec AD transistor)

개인 학습 프로젝트입니다. 펨트론 장비·고객 데이터와 무관하며, 이미지는 [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) 공식 페이지에서 받습니다.

**License:** MVTec AD — CC BY-NC-SA 4.0 (비상업적 사용). 슬라이드 예시에는 *MVTec AD, Bergmann et al., CC BY-NC-SA 4.0* 을 표기합니다.

## 데이터 배치

압축과 원본 PNG는 Git에 올리지 않습니다. `transistor` 폴더를 저장소 밖에 두고 환경 변수로 경로를 지정합니다.

```powershell
$env:TRANSISTOR_DATA_ROOT = "D:\data\mvtec_ad\transistor"
```

기대 구조: `train/good`, `test/good`, `test/bent_lead`, …, `ground_truth/<유형>/`.

## 설치

```powershell
pip install -r requirements.txt
```

이 저장소에서 학습과 추론에 쓴 환경은 Python 3.12.7, PyTorch 2.12.1+cpu, 로컬 CPU입니다.

## 실행 순서

```powershell
# 1일: 장수 확인, split, 예시 그림
python data.py --write-split --examples

# 2일: AE 학습 + 방법 A 점수 (GPU는 Kaggle 노트북에서 동일 스크립트 실행 가능)
python train_autoencoder.py

# Kaggle (train/good 213장만): 로컬 split.json을 함께 올린 뒤
python train_autoencoder.py --train-only --split-file outputs/split.json
# 세션 종료 전 autoencoder.pt, ae_loss.csv 다운로드 → 로컬 outputs/
python train_autoencoder.py --scores-only

# 3일: 방법 B 점수 + bank.pt
python score_embedding.py

# 4일: 문턱값, summary, 그림 (재학습 없음)
python evaluate.py

# 5일: 밝기 + 추론 시간 + summary 갱신
python score_brightness.py
python evaluate.py --timing --finalize
```

Kaggle에서 학습할 때는 `train/good` 213장만 비공개 데이터셋으로 올리고, 로컬에서 만든 `outputs/split.json`과 스크립트를 그대로 실행합니다. 세션 종료 전 `outputs/autoencoder.pt`와 `outputs/ae_loss.csv`를 로컬 `outputs/`로 받습니다.

## 산출물

`outputs/split.json`, `autoencoder.pt`, `ae_loss.csv`, `bank.pt`, `scores_raw.csv`, `scores.csv`, `summary.md`, `figures/`.
