# Privacy and responsible use

Contract documents can contain privileged, confidential, personal, and trade
secret information. AI-Enabled Contracts Lab therefore uses local-only defaults.

## Data flow

- The Streamlit server runs on the local computer.
- Uploaded contract bytes are parsed in memory.
- Training datasets are only written when the user chooses a save action.
- Model downloads contact the selected model host.
- The optional CUAD action contacts Hugging Face to obtain public data.
- Weights & Biases is disabled until the user opts in.

When W&B is enabled, AI-Enabled Contracts Lab logs scalar metrics and non-sensitive training
configuration. It does not log contract text, filenames, raw predictions, or
dataset tables. The user remains responsible for their W&B account settings and
organizational policies.

## Responsible interpretation

AI-Enabled Contracts Lab separates extracted evidence, model classification, heuristic
interpretation, and playbook comparison. None of those layers establishes a
legal conclusion. Confidence values measure model behavior, not legal certainty.

Do not use the system as a substitute for a lawyer, for autonomous approval or
rejection of agreements, or for decisions affecting a person's legal rights.
