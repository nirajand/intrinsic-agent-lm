# Agent instructions

Do not claim frontier capability from a small checkpoint. Keep architecture, data, optimization, and evaluation claims separate.

Every new Python module requires a unit test. Any dataset adapter must document the source schema and output schema. Any new training objective must include its mathematical definition, a reference, and a numerical smoke test.

Never silently convert streaming datasets into full local downloads during a training run. Pin dataset revisions for reproducibility. Keep human-feedback, AI-feedback, verifier, and model-self-generated signals distinct in manifests and metrics.

A `.safetensors` file stores tensor values and informational metadata. It must not be described as containing executable runtime behavior.
