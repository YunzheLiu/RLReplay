# RLReplay
Code accompanying the paper: Liu, Y., Mattar, M., Behrens, T., Daw, N., & Dolan, R. J. (2020). Experience replay supports non-local learning. bioRxiv.

## Simulation: replay strength vs. state decoding accuracy
run ``` Simulation_SeqStrength_vs_StateDecode.m ```

## Computational Modeling
for prioritization in non-local learing:

based on need - ``` NonLocal_Learning_Need.stan ```
based on gain - ``` NonLocal_Learning_Gain.stan ```

for replay promote RL model:
run ``` Learning_Replay.stan ```
