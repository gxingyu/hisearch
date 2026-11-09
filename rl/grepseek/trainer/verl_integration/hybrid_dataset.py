from __future__ import annotations

import json
from pathlib import Path

from grepseek.prompting import PromptAssets, render_system_prompt
from grepseek.trainer.verl_integration.dataset import GrepSeekDataset
from inference.retrieval_env.schemas import tool_schemas


def build_hybrid_system_prompt(config) -> str:
    assets = PromptAssets.from_files(
        system_prompt_path=Path(config["system_prompt_path"]),
        retrieval_instruction_path=Path(config["retrieval_instruction_path"]),
        tool_spec_template_path=Path(config["tool_spec_template_path"]),
    )
    return render_system_prompt(
        assets=assets, tool_spec=json.dumps(tool_schemas(), ensure_ascii=False, separators=(",", ":"))
    )


class HybridDataset(GrepSeekDataset):
    def _build_system_prompt(self, config) -> str:
        return build_hybrid_system_prompt(config)


class NativeHybridDataset(HybridDataset):
    """Use a native tool template instead of duplicating JSON examples/schema."""

    def _build_system_prompt(self, config) -> str:
        return Path(config["system_prompt_path"]).read_text(encoding="utf-8").strip()
