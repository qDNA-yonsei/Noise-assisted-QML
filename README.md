# Noise-Assisted Annealing (Adaptive Pauli)

Pauli 노이즈를 annealing schedule로 활용해 VQE barren plateau / local trap을 탈출하는 방법을 비교하는 실험 코드.

---

## 파일 구조

```
260319/
├── config.py               # 모든 하이퍼파라미터
├── hamiltonian.py          # H = Σ Z_i
├── ansatz.py               # StronglyEntanglingLayers (clean + Pauli-noise 변형)
├── noise.py                # cost function 팩토리 (clean / Pauli / shot)
├── utils.py                # 파라미터 초기화, optimizer, Hessian 진단, 수렴 판정
├── result.py               # training mode별 dataclass
├── training/
│   ├── seed_search.py      # [Mode 1] seed search → SeedSearchResult
│   └── run.py              # [Mode 2/3/4] clean / pauli / shot 실험
└── main.py                 # 진입점: run_all()
```

---

## 데이터 흐름

```
main.py::run_all()
    │
    ├─ [Mode 1] training/seed_search.py::run_seed_search()
    │       1000개 seed → clean optimizer → trapped seed 선별
    │       └─ SeedSearchResult
    │               ├─ selected: SeedDiagnosis   ← 대표 trapped seed
    │               └─ all_problematic: list[SeedDiagnosis]
    │
    ├─ [Mode 2] training/run.py::run_pauli_annealing(init_params)
    │       Pauli schedule [0.4 → 0.3 → ... → 0.0] 단계적 noise 감소
    │       └─ PauliAnnealingResult
    │               ├─ stages: list[PauliStageResult]  ← stage별 수렴 정보
    │               ├─ history / history_steps          ← 전체 학습 곡선
    │               └─ final_clean_eval, final_grad_norm, final_lambda_min
    │
    ├─ [Mode 3] training/run.py::run_clean(init_params, steps=budget)
    │       Pauli와 동일한 step budget으로 noiseless 최적화
    │       └─ CleanRunResult
    │               ├─ eval_hist, final_eval
    │               └─ grad_norm, lambda_min, lambda_max
    │
    └─ [Mode 4] training/run.py::run_shot_experiments(init_params, steps, clean_final)
            shots=[1000, 500, 100, 50] × 10 repeats
            └─ list[ShotRunResult]
                    ├─ repeats: list[ShotRepResult]
                    ├─ escape_count / num_repeats
                    ├─ best_rep: ShotRepResult   ← min_clean_eval 기준
                    └─ stuck_rep: ShotRepResult  ← 탈출 못한 대표 rep
```

---

## result.py 구조

```
ExperimentResult
├─ init_seed: int
├─ seed_search: SeedSearchResult | None
├─ seed_diagnosis: SeedDiagnosis
├─ clean_run: CleanRunResult
├─ pauli_run: PauliAnnealingResult
│     └─ stages: list[PauliStageResult]
└─ shot_runs: list[ShotRunResult]
      └─ repeats: list[ShotRepResult]
```

---

## 주요 설정 (config.py)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `N_QUBITS` | 4 | 큐비트 수 |
| `N_LAYERS` | 2 | SEL layer 수 |
| `CLEAN_OPTIMIZER` | `"adam"` | `"gd"` / `"adam"` / `"spsa"` |
| `RUN_SEED_SEARCH` | `True` | False이면 `INIT_SEED` 고정 사용 |
| `SEARCH_SEEDS` | `range(1000)` | seed 탐색 범위 |
| `PAULI_SCHEDULE` | `[0.4 … 0.0]` | noise annealing schedule |
| `SHOT_LIST` | `[1000,500,100,50]` | 비교할 shots 값들 |

---

## 실행

```bash
cd /home/yujin/noise_assisted_annealing/260319
conda activate noise_qml

# 기본 실행 (config.py 기본값 그대로)
python main.py

# 도움말
python main.py --help
```

### CLI 인자 (모두 선택적, 기본값은 config.py)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--n-qubits` | 4 | 큐비트 수 |
| `--n-layers` | 2 | SEL layer 수 |
| `--no-seed-search` | (flag) | seed search 생략, `--init-seed` 사용 |
| `--init-seed` | 1 | 고정 seed (--no-seed-search 시 사용) |
| `--search-n` | 1000 | 탐색할 seed 개수 |
| `--search-steps` | 500 | seed당 optimizer step 수 |
| `--optimizer` | adam | 전체 모드 optimizer 한번에 설정 (`gd`/`adam`/`spsa`) |
| `--lr-adam` | — | 전체 모드 adam lr 한번에 설정 |
| `--clean-optimizer` | adam | clean mode optimizer |
| `--pauli-optimizer` | adam | pauli mode optimizer |
| `--shot-optimizer` | adam | shot mode optimizer |
| `--lr-clean-adam` | 0.01 | |
| `--lr-pauli-adam` | 0.01 | |
| `--lr-shot-adam` | 0.01 | |
| `--pauli-schedule` | 0.4 0.3 … 0.0 | noise schedule (공백 구분) |
| `--pauli-min-steps` | 50 | |
| `--pauli-max-steps` | 1000 | |
| `--shot-list` | 1000 500 100 50 | 테스트할 shots 값 (공백 구분) |
| `--shot-repeats` | 10 | |
| `--save-prefix` | adaptive_pauli_vs_shot_sel | 출력 파일명 prefix |

### 자주 쓰는 예시

```bash
# 빠른 테스트 (전체 파이프라인 확인용)
python main.py \
  --search-n 10 \
  --search-steps 50 \
  --pauli-max-steps 100 \
  --shot-repeats 2

# seed search 생략하고 특정 seed로 바로 실험
python main.py --no-seed-search --init-seed 42

# 큐비트 수 늘리기
python main.py --n-qubits 6 --n-layers 3

# optimizer / lr 바꾸기
python main.py --optimizer gd --lr-clean-gd 0.05

# shot 조합 바꾸기
python main.py --shot-list 500 100 --shot-repeats 5

# 저장 prefix 지정
python main.py --save-prefix my_exp_01
```

### 출력 파일

실행 후 현재 디렉토리에 생성됨:

| 파일 | 내용 |
|------|------|
| `*_summary.json` | 전체 결과 요약 (JSON) |
| `*_comparison.png` | 3패널 비교 그래프 |
| `*_clean_eval_hist.npy` | clean 학습 곡선 |
| `*_pauli_hist.npy` | Pauli annealing 체크포인트 곡선 |
| `*_pauli_final_params.npy` | Pauli annealing 최종 파라미터 |
| `*_shot_{N}_best_eval_hist.npy` | shot noise best rep 곡선 |

---

## GPU 최적화 옵션

> 현재 코드는 PennyLane `autograd` 인터페이스 (CPU). 아래 두 가지 경로로 GPU 가속 가능.

### Option A — `lightning.gpu` (권장, 코드 변경 최소)

cuQuantum 기반 GPU 시뮬레이터. `noise.py`의 device만 교체.

```bash
pip install pennylane-lightning[gpu]   # cuQuantum 필요
```

```python
# noise.py
dev = qml.device("lightning.gpu", wires=N_QUBITS)   # clean cost
# diff_method → "adjoint" 권장 (parameter-shift보다 빠름)
```

> **주의**: `default.mixed` (Pauli noise stage)는 GPU 미지원.
> Pauli stage는 CPU로 fallback되거나 별도 처리 필요.

### Option B — JAX + GPU backend (더 큰 속도 향상 가능)

```bash
pip install "jax[cuda12]"
```

`noise.py`, `utils.py` 전반의 `interface="autograd"` → `interface="jax"` 변경,
`pennylane.numpy` → `jax.numpy`로 교체, `jax.jit`으로 cost function 컴파일.

```python
import jax
import jax.numpy as jnp

@jax.jit
@qml.qnode(dev, interface="jax", diff_method="backprop")
def cost(weights):
    apply_sel(weights)
    return qml.expval(H)
```

### 현실적 판단

| N_QUBITS | GPU 효과 |
|----------|---------|
| 4–10 | 거의 없음 (overhead > gain) |
| 12–18 | 의미있는 speedup |
| 20+ | GPU 필수 |

현재 N_QUBITS=4에서는 GPU보다 **seed search 병렬화** (`multiprocessing`)가 더 실용적:

```python
from multiprocessing import Pool

with Pool(processes=8) as pool:
    diagnoses = pool.map(diagnose_seed, SEARCH_SEEDS)
```
