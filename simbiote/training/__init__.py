"""BC pretrain -> PPO fine-tune -> play -> export, per spec §5.4.

Build order (per the doc): policy_net -> bc_pretrain -> train_nav ->
train_grasp -> play -> export_policy. `retrain.py` wires bc_pretrain +
train_* together behind the OpenClaw `ingest_demo()`/`finetune_policy()`
tool calls (Part 1's orchestration table).
"""
