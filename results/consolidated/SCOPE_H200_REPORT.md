# SCOPE H200 실행 결과 — 2026-10-18 KST

상태: 초기 연구 캠페인 실행 완료 / 전체 연구 범위 PARTIAL. OSDI 수준 실험 완성, 신규성 또는 시스템 수준 성능 개선은 확립되지 않았다.

## 공개 결과·고정 버전

- 저장소: https://github.com/moonhyunseung/SCOPE/tree/research/h200-pilot-20261018
- 검증된 결과 커밋: 9de5bacb2c34282bd8e821f99010401c9a78ec83
- 초기 캠페인 7141e594975ea70c: 7d3c442bc1c3e154d64b274c830cea522da391c1
- 크기 확장 479dde6421f18d68: ba0b4bbdea12be44aa7b6686c766e6173da0e85a
- 공통 실행 소스 SHA256: 2bdb2195a5c58aa1ba05efdf820a90e056ac3649ad9ef8c1f50bc932e5db1cab
- 확장 프로토콜의 원래 로컬 커밋도 Git bundle에 보존했다. 서버 push 인증 전에는 일부 로컬 커밋이 원격에서 직접 조회되지 않을 수 있다.

## 환경과 실행

NVIDIA H200 1대, nvidia-smi HBM 143771 MiB, driver 580.126.09. Python 3.12.3, PyTorch 2.14.1+cu130(메타데이터 2.14.1), CUDA runtime 13.0, Triton 3.8.0, Transformers 5.19.0, NumPy 2.5.4, pytest 9.0.2, huggingface_hub 2.2.0. FlashAttention/vLLM 미설치. Triton import만 확인했으며 별도 Triton 커널을 측정하지 않았다.

별도 SCOPE checkout/venv에서 기존 CUDA 패키지를 읽어 재사용했다. 기존 패키지 버전은 교체하지 않았다. 초기 GPU 프로세스 없음과 H200 장치를 확인했다. PyTorch 할당 한도 96 GiB, HBM 안전 여유 12 GiB, 디스크 여유 24 GiB, 조건별·캠페인별 시간 제한을 적용했다. tmux, SQLite, 원자적 쓰기, heartbeat, 중복 실행 잠금을 사용했다. 실험 worker는 LLM API 없이 독립 실행했다.

## 계획 조건과 실패 분모

수정된 두 캠페인: {"PASS": 34, "UNSUPPORTED": 4, "SKIPPED_WITH_REASON": 24, "FAIL": 8} / 총 70개 실행 블록. 이 숫자에는 검증·집계·미지원·건너뜀도 포함된다. 모델 조건 20개 중 PASS 8, FAIL 8, UNSUPPORTED 4. 단일 레이어 연산 후보 32개 중 PASS 24, FAIL 8이며 미통과 후보는 timing을 금지하되 등록부에는 남겼다.

별도 초기 v1 디버깅 실행 41개 블록(통과 11, 실패 6, 미지원 4, 건너뜀 20)은 BF16 모델을 FP32 진단 참조로 잘못 게이트한 이력이다. 수정된 결과 통계와 섞지 않았으며 원본 결과와 Git 이력을 별도 패키지에 보존했다. 총 처리 이력 111개 블록은 프로토콜 간 중복과 집계 블록을 포함한다.

| 모델 | revision | PASS | FAIL | 미지원 |
|---|---|---:|---:|---:|
| Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | 4 | 0 | 0 |
| EleutherAI/pythia-1b | f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2 | 2 | 0 | 2 |
| facebook/opt-1.3b | 3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 | 0 | 2 | 2 |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | 0 | 4 | 0 |
| Qwen/Qwen2.5-14B-Instruct | cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 | 2 | 2 | 0 |

초기 길이 512/2048, batch 1, 고정 공개 반복 prose/code, seed 20261018, 첫·중간·마지막 attention 레이어, 3 decode 위치를 사용했다. 실제 Q/K/V와 eager가 반환한 P/Y를 수집했다. FP32 Z/P/Y/W 및 예시 quantizer 코드·스케일은 재구성 진단값이다. native fused-kernel 상태로 기술하지 않았다. 통과 조건의 1296개 head/step/layer 관측은 독립 표본이 아니다.

## 검증·실측 결과

테스트 14개 통과. 합성 MHA/GQA/MQA, 마스크, append, packing, runner 무결성, 원래 모델 함수와 동일 dtype replay를 검사했다. 실제 모델 통과 조건의 동일 BF16 연산 순서 replay attention relative L2는 0이었다. BF16 QK/scale/mask, FP32 softmax 후 BF16 cast, BF16 PV와 FP32 진단의 오차를 분리했다. 기존 실패를 통과시키려고 허용 오차를 올리지 않았다.

OPT-1.3B 2개, Qwen-7B 4개, Qwen-14B 2개 조건은 cached/full logits relative L2 0.025 게이트에 실패했다. 원인은 미확정이다. 원래 실패 출력에 정확한 오차 크기가 저장되지 않아 UNKNOWN이며 정밀도·캐시 구현 중 하나라고 단정하지 않는다. 이 조건의 후속 성능 결론을 차단했다. Pythia/OPT의 2048 조건은 추가 decode가 위치 한도를 넘으므로 미지원이다.

캡처된 첫 레이어 입력 8개에서 packed K/V와 FP32 group scale의 저장 텐서는 dense BF16보다 3.20배 작았다. unpack+dequantize+eager 전체 경로는 dense eager보다 2.70–2.82배 느렸다. 일회성 packing 비용은 별도 기록했다. 워크스페이스도 측정했지만 dense·참조 텐서가 함께 남은 제어된 replay이다. 저장 비율을 최대 resident KV 용량으로 해석할 수 없다.

Qwen-1.5B의 SDPA/packed+SDPA 후보 8개는 이 입력과 BF16 eager 수치 계약에 실패해 timing을 금지했다. 이는 SDPA의 일반적인 버그나 열등함을 뜻하지 않는다. 예시 INT4의 end-model 품질은 검증하지 않았다.

## Full-model dense baseline

| 모델 | 초기 L | 입력 | decode step 중앙값 ms |
|---|---:|---|---:|
| Qwen/Qwen2.5-1.5B-Instruct | 512 | prose | 13.2692 |
| Qwen/Qwen2.5-1.5B-Instruct | 512 | code | 13.2767 |
| Qwen/Qwen2.5-1.5B-Instruct | 2048 | prose | 14.1290 |
| Qwen/Qwen2.5-1.5B-Instruct | 2048 | code | 13.6436 |
| EleutherAI/pythia-1b | 512 | prose | 6.2933 |
| EleutherAI/pythia-1b | 512 | code | 5.4746 |
| Qwen/Qwen2.5-14B-Instruct | 512 | code | 25.0375 |
| Qwen/Qwen2.5-14B-Instruct | 2048 | prose | 24.7628 |

위 값은 instrumentation을 제거한 eager BF16 full-model forward와 greedy token 선택의 CUDA-event 구간이다. 3개 반복 창, 창마다 timed token 6개. prefill·detokenization·요청 전송·serving overhead는 제외한다. 문맥은 decode마다 증가한다. 커널 측정과 합쳐 end-to-end 수치를 만들지 않았다.

## 게시와 무결성

수정된 70개 블록의 manifest/summary/CSV/report/SHA256SUMS 총 350개를 원격 Git 커밋에서 읽어 H200 원본 바이트·SHA256과 일치함을 확인했다. 두 캠페인의 reproduce 명령은 ARTIFACT_INTEGRITY_PASS였다. 독립 GPU 재실행은 아니다. 내려받은 패키지의 존재하는 파일 406개도 해시 불일치가 없었다. 생략된 원시 파일 64개는 H200 관리 디렉터리에 남는다.

게시 중 CSV 줄바꿈 정규화로 해시가 달라진 중간 커밋은 원본 바이트 보존 커밋으로 수정하고 재검증했다. 기존 이력은 삭제하지 않았다. 개별 보고서의 PENDING은 내보내기 당시 스냅샷이다. 최종 70개는 연결된 GitHub 도구로 게시하고 서버에서 원격 파일을 검증한 뒤 PUBLISHED로 기록했다. 서버 자체 git push는 인증 부재로 exit 128이며 자동 push 성공으로 간주하지 않는다.

## 재현

Git bundle에서 새 checkout을 만든다. 기록된 CUDA 환경을 별도 venv에 재구성하고 H200 gate를 통과해야 한다.

~~~bash
git clone scope_v2_history.bundle SCOPE-reproduce
cd SCOPE-reproduce
git checkout 7d3c442bc1c3e154d64b274c830cea522da391c1
python -m scope.runner preflight --config configs/h200_campaign.yaml
python -m scope.runner plan --config configs/h200_campaign.yaml
python -m scope.runner run --config configs/h200_campaign.yaml --resume --publish-each-block
~~~

크기 확장은 별도 새 checkout에서 ba0b4bbdea12be44aa7b6686c766e6173da0e85a를 사용하고 config를 configs/h200_size_extension.yaml로 바꾼다. 이전 results/를 복사하지 않는다. 모델 revision·입력·설정은 campaign.json에도 동결되어 있다. 원래 결과 무결성은 H200 원본 디렉터리에서 다음으로 검사한다.

~~~bash
python -m scope.runner reproduce --run-id 7141e594975ea70c
python -m scope.runner reproduce --run-id 479dde6421f18d68
~~~

서버 Python: /home/inugpu/workspace/scope_h200_20261018_9EfrwM/SCOPE/.venv/bin/python. 작업 디렉터리: /home/inugpu/workspace/scope_h200_20261018_9EfrwM/SCOPE_v2. 원시 tensor/log는 results/campaigns/<run-id>/에 남아 있다. 모델 가중치·인증 정보·원시 tensor는 GitHub에 게시하지 않았다.

## 남은 연구 결정

- resident-KV 최대 용량, 페이지/패딩/메타데이터 포함 serving frontier: 미측정.
- 실제 모델 compressed-cache 통합, end-model 품질/perplexity: 미측정.
- 실제 MQA, sliding/hybrid/MLA, FlashAttention/vLLM, custom Triton·직접 packed·TMA/MMA lowering: 미검증.
- 긴 문맥·batch 2/4/8, held-out factorial confirmation, 독립 GPU replication, traffic/occupancy profiling: 미실행.
- cached/full logits 실패의 정확한 오차와 dtype/position/cache 원인을 별도 진단한 뒤 모델 fidelity 계약을 확정해야 한다.
- 그 다음에 tile-local 복원과 dense 복원 등 경쟁 경로를 비교해야 한다. 현재 자료로 하드웨어 경로의 우승이나 OSDI 수준 완성을 주장할 수 없다.
