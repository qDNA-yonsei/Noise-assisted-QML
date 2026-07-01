# commeff_tfim_8qubit — High Noise Schedule Experiment Plan

## Overview

이 실험은 `commeff_tfim_8qubit_20260624_182713` (이하 **기준 실험**)과 **완전히 동일한 조건**에서
noise annealing schedule의 강도만 2배로 높여 그 효과를 직접 비교하기 위한 통계 실험이다.

| 항목 | 기준 실험 | **이 실험 (high noise)** |
|------|-----------|--------------------------|
| Importance scoring | commutator_eff (n_avg=50) | 동일 |
| Hamiltonians | H00000–H00004 | 동일 |
| Seeds | 0–99 (100개) | 동일 |
| Circuit | 8-qubit, 4-layer, 96 gates | 동일 |
| Methods | 7개 (clean + top/bottom 10/50/90%) | 동일 |
| Noise schedule (YJ) | [0.4 → 0.0] | **[0.8 → 0.0]** |
| Noise schedule (jungyun 환산) | [0.8 → 0.0] | **[1.6 → 0.0]** |

**연구 목적**: noise schedule 강도가 VQE 수렴 및 최종 에너지에 미치는 영향 정량화.
기준 실험과 이 실험의 데이터를 직접 비교하여 schedule의 효과를 분리한다.

---

## Circuit

| Parameter | Value |
|-----------|-------|
| n_qubits  | 8 |
| n_layers  | 4 |
| ranges    | [1, 2, 3, 4] |
| total rotation gates | 96 |
| entangling | CNOT (StronglyEntanglingLayers) |

---

## Hamiltonians

5 fixed TFIM Hamiltonians (기준 실험과 동일):

`H = -jzz * Σ Z_i Z_{i+1} - hx * Σ X_i`

| ID     | jzz      | hx       | ground energy |
|--------|----------|----------|---------------|
| H00000 | -1.02277 |  1.04293 | -10.18344 |
| H00001 |  0.91856 | -1.91341 | -16.08721 |
| H00002 | -1.92244 |  0.84161 | -14.39433 |
| H00003 |  1.32125 |  0.80994 | -10.54559 |
| H00004 | -1.71190 |  0.17443 | -12.02779 |

---

## Seeds

- 0 to 99 (100 seeds total)
- Initialization: `np.random.default_rng(seed).uniform(0.0, 2π, (n_layers, n_qubits, 3))`
  (기준 실험과 동일)

---

## Importance Scoring: commutator_eff

기준 실험과 동일 (변경 없음):

```
O ← H_mat
for gate k (reversed):
    score[k] = ||[O, G_k]||_F
    O ← U_k† O U_k
```

`n_avg=50` random parameter sets 평균.

---

## Noise Model

기준 실험과 동일:

- **Mode**: matched Pauli noise
- **Application**: `PauliError("Z", p)` before RZ gates, `PauliError("Y", p)` before RY gates
- **Convention**: YJ는 `p`를 `qml.PauliError`에 직접 전달 (p/2 변환 없음)
- **Equivalence**: YJ schedule × 2 = jungyun schedule (동일한 물리적 noise 강도)

---

## Annealing Schedule

이 실험의 핵심 변경점. 기준 실험 대비 **모든 p값 2배**.

| 단계 | 기준 실험 (YJ) | **이 실험 (YJ)** | jungyun 환산 |
|------|----------------|------------------|--------------|
| 1    | 0.4            | **0.8**          | 1.6          |
| 2    | 0.3            | **0.6**          | 1.2          |
| 3    | 0.2            | **0.4**          | 0.8          |
| 4    | 0.1            | **0.2**          | 0.4          |
| 5    | 0.05           | **0.1**          | 0.2          |
| 6    | 0.025          | **0.05**         | 0.1          |
| 7    | 0.01           | **0.02**         | 0.04         |
| 8    | 0.005          | **0.01**         | 0.02         |
| 9    | 0.0            | 0.0              | 0.0          |

```python
# sweep_8qubit.py 에서 변경할 상수
PAULI_SCHEDULE = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
```

Per stage 수렴 조건 (기준 실험과 동일):
- `PAULI_MIN_STEPS  = 30`
- `PAULI_MAX_STEPS  = 500`
- `PAULI_CHECK_EVERY = 10`
- `PAULI_WINDOW     = 20`
- `PAULI_STD_TOL    = 0.005`
- `PAULI_RATE_TOL   = 0.005`

---

## Methods (7개, 기준 실험과 동일)

| # | method_key | description |
|---|-----------|-------------|
| 1 | `clean` | No noise, 1000 fixed steps (baseline) |
| 2 | `top10_fixed` | commutator_eff top 10% (~10 gates) noisy |
| 3 | `top50_fixed` | commutator_eff top 50% (~48 gates) noisy |
| 4 | `top90_fixed` | commutator_eff top 90% (~86 gates) noisy |
| 5 | `bottom10_fixed` | commutator_eff bottom 10% noisy |
| 6 | `bottom50_fixed` | commutator_eff bottom 50% noisy |
| 7 | `bottom90_fixed` | commutator_eff bottom 90% noisy |

---

## 실행 방법

### 1. sweep_8qubit.py 수정

`PAULI_SCHEDULE` 상수 한 줄만 변경:

```python
# 변경 전 (기준 실험)
PAULI_SCHEDULE = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0]

# 변경 후 (이 실험)
PAULI_SCHEDULE = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
```

### 2. 출력 디렉토리

```
D:\yujin\commeff_tfim_8qubit_<YYYYMMDD_HHMMSS>\
```

기준 실험과 동일한 구조, 타임스탬프만 다름.

### 3. 실행 스크립트 (기준 실험과 동일)

```
python run_h_seeds.py 0 D:\yujin\commeff_tfim_8qubit_<timestamp>
python run_h_seeds.py 1 D:\yujin\commeff_tfim_8qubit_<timestamp>
python run_h_seeds.py 2 D:\yujin\commeff_tfim_8qubit_<timestamp>
python run_h_seeds.py 3 D:\yujin\commeff_tfim_8qubit_<timestamp>
python run_h_seeds.py 4 D:\yujin\commeff_tfim_8qubit_<timestamp>
```

5개 프로세스 동시 실행.

---

## Output Format

기준 실험과 완전히 동일:

```
commeff_tfim_8qubit_<timestamp>/
├── H00000/
│   └── seed00000/
│       ├── config.json
│       ├── eff_scores.json
│       ├── clean_e.npy / clean_steps.npy
│       ├── top10_fixed_e.npy / top10_fixed_steps.npy
│       ├── ...
│       ├── detailed_results.csv
│       └── run.log
├── H00001/ ...
└── ...
```

---

## 비교 실험 구조

| 실험 | 디렉토리 | PAULI_SCHEDULE (YJ) | 목적 |
|------|----------|---------------------|------|
| 기준 실험 | `commeff_tfim_8qubit_20260624_182713` | [0.4 → 0.0] | baseline |
| **이 실험** | `commeff_tfim_8qubit_<new_timestamp>` | **[0.8 → 0.0]** | high noise 비교 |

두 실험의 `detailed_results.csv`를 seed별로 paired comparison하여 schedule 효과 분석.

---

## Status

- [ ] `sweep_8qubit.py` `PAULI_SCHEDULE` 수정
- [ ] 출력 디렉토리 생성 및 실행
- [ ] 완료 후 기준 실험과 paired comparison 분석
