This patched version keeps the public interface simple by always using the
single name `p`, while still making the matched-noise path align exactly with:
"Regularizing quantum loss landscapes by noise injection".

Key changes
-----------
1. Exact matched channel with a unified public parameter name:
   - You always pass `p` from the CLI/config.
   - In matched mode, that user-facing `p` is interpreted as the paper's
     regularization strength and is implemented internally as
         qml.PauliError(P, p/2)
     so the paper channel is reproduced exactly.
   - In layerwise mode, the same user-facing `p` is used directly as the
     PauliError probability.

2. Range enforcement:
   - user-facing p is always restricted to [0, 1]
   Inputs outside the valid interval now raise an error instead of being clipped.

3. Two schedule modes are available:
   - manual:
       a directly specified stage schedule (PAULI_SCHEDULE / --pauli-schedule)
       with convergence-based transitions, like the original project flow.
   - fixed_step:
       the paper-style continuously decaying schedule
       p(i) = p_max * exp(-a * i / i_max)
       over a fixed number of optimization steps.

4. Default schedule mode:
   manual

5. Logging/summaries now record the user-facing p consistently,
   while comments in the code explain the matched-mode internal p/2 mapping.

Typical matched-mode commands
-----------------------------
Manual schedule (default)
python main.py   --n-qubits 8   --no-seed-search   --init-seed 51   --noise-mode matched   --pauli-schedule 0.8 0.6 0.4 0.2 0.1 0.06 0.02 0.014 0.01 0.0

Paper-style fixed-step schedule
python main.py   --n-qubits 8   --no-seed-search   --init-seed 51   --noise-mode matched   --schedule-mode fixed_step   --p-max 0.9   --noise-decay-a 10   --pauli-total-steps 2000
