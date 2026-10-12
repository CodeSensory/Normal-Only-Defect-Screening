# Summary

## 기기와 버전
- Python: 3.12.7
- PyTorch: 2.12.1+cpu
- 학습 기기 (AE): cpu
- 추론 기기: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- AE 학습 파라미터 수: 1117955
- ResNet-18 학습 파라미터 수: 0 (고정 특징)

## 문턱값 (검증 95%)
- AE: 0.002250
- ResNet-18: 0.783784

## 혼동행렬 (양성=불량)
- AE: TN=58 FP=2 FN=29 TP=11
- ResNet-18: TN=59 FP=1 FN=5 TP=35

## 오탐과 미탐
- AE: 오탐 2장, 미탐 29장
- ResNet-18: 오탐 1장, 미탐 5장

## 유형별 재현율 (맞힌 장수 / 10)
- Bent lead: AE 3/10, ResNet-18 10/10
- Cut lead: AE 0/10, ResNet-18 8/10
- Damaged case: AE 0/10, ResNet-18 9/10
- Misplaced: AE 8/10, ResNet-18 8/10

## AUROC (테스트 100장)
- AE: 0.6733
- ResNet-18: 0.9500

## 추론 시간 (배치 1, 이미지당 ms)
- autoencoder / forward: 11.125 ms (Intel64 Family 6 Model 142 Stepping 12, GenuineIntel)
- embedding / feature_extract: 39.896 ms (Intel64 Family 6 Model 142 Stepping 12, GenuineIntel)
- embedding / nearest_distance: 4.931 ms (Intel64 Family 6 Model 142 Stepping 12, GenuineIntel)

## 밝기
- 배율 0.7, AE: 정상 60장 중 판정 변경 56장, 불량 40장 중 판정 변경 28장
- 배율 0.7, ResNet-18: 정상 60장 중 판정 변경 1장, 불량 40장 중 판정 변경 1장
- 배율 1.3, AE: 정상 60장 중 판정 변경 58장, 불량 40장 중 판정 변경 29장
- 배율 1.3, ResNet-18: 정상 60장 중 판정 변경 1장, 불량 40장 중 판정 변경 2장
