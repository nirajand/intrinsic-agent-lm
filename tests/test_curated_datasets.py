from pathlib import Path
import json
import yaml

from ialm.data import DatasetRegistry, DatasetSpec
from ialm.processors import pretrain, sft, scientific_sft, code_task_sft, rlvr, tool_trace


def test_curated_registry_and_recipe():
    catalog = json.loads(Path("data/catalog.json").read_text())
    expected = {
        "arxiv-complete-paper-text",
        "zgcm-1-pretrain-stage1",
        "ultradata-code-l3-python",
        "ultradata-sft-agent-code",
        "ultradata-sft-agent-tool-use",
        "fable-5-premium-v2",
        "morethought-fable-5-1-reasoning",
        "spark-234k-scientific-reasoning",
        "ultradata-rl-math",
        "ultradata-rl-code",
        "kimi-cyber-reasoning",
        "oi-uae-cyber-security",
        "saidutta69-fable-5-premium-v1-legacy",
    }
    assert expected <= set(catalog)
    recipe = yaml.safe_load(Path("configs/recipe_full.yaml").read_text())
    repos = {d["repo"] for stage in recipe["datasets"].values() for d in stage}
    assert "secemp9/arxiv-complete" in repos
    assert "openbmb/UltraData-SFT-Agent-2609" in repos
    assert "openbmb/UltraData-RL-2609" in repos
    assert "zgcagi/ZGCM-1-Data" in repos


def test_new_processors():
    assert pretrain({"paper_text": "A paper"}) is None
    assert pretrain({"paper_text": "A scientific paper with enough text to train."})["text"]
    assert sft({"instruction": "Solve this.", "output": "Solution."})["messages"][-1]["role"] == "assistant"
    assert scientific_sft({"instruction": "Explain", "output": "Because."})["messages"][-1]["content"] == "Because."
    assert code_task_sft({"task": "Write f", "analysis": "Reason", "solution": "def f(): pass"})["messages"][-1]["content"].startswith("Reason")
    assert rlvr({"query": "2+2", "ground_truth": "4"})["answer"] == "4"
    row = tool_trace({"messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"call_1"}]}]})
    assert row["messages"][0]["tool_calls"][0]["id"] == "call_1"
