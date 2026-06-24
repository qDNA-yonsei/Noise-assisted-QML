#!/bin/bash
# H00000, seed=0 에서 clean vs full 비교 테스트
# commutator_eff_YJ_260521/ 폴더에서 실행

cd "$(dirname "$0")"
python test_clean_vs_full.py --max_steps 300
