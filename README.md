# Noise-Assisted Annealing (Adaptive Pauli)

## 구조

```
.
├── main.py                     # 진입점
├── src/
│   ├── config.py               # 하이퍼파라미터
│   ├── hamiltonian.py          # Hamiltonian 정의
│   ├── ansatz.py               # Ansatz 회로
│   ├── noise.py                # Cost function (clean / Pauli / shot)
│   ├── utils.py                # 유틸리티
│   ├── result.py               # 결과 dataclass
│   ├── args.py                 # CLI 인자 파싱
│   └── training/
│       ├── seed_search.py      # Seed 탐색
│       └── run.py              # 실험 실행
├── outputs/                    # 실험 결과 저장 (git 제외)
└── tests/                      # 테스트
```
