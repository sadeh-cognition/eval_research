"""HoVer multi-hop claim verification data loading."""

from __future__ import annotations

import random

import dspy
from dspy.datasets import DataLoader


def load_hover(
    *,
    train_size: int = 100,
    dev_size: int = 100,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Load 3-hop HoVer examples and split into train/dev/test.

    Matches the official tutorial:
    https://dspy.ai/tutorials/agents/
    """
    kwargs = dict(
        fields=("claim", "supporting_facts", "hpqa_id", "num_hops"),
        input_keys=("claim",),
    )
    hover = DataLoader().from_huggingface(
        dataset_name="vincentkoc/hover-parquet",
        split="train",
        **kwargs,
    )

    hpqa_ids: set = set()
    examples = [
        dspy.Example(
            claim=x.claim,
            titles=list({y["key"] for y in x.supporting_facts}),
        ).with_inputs("claim")
        for x in hover
        if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
    ]

    random.Random(seed).shuffle(examples)
    trainset = examples[:train_size]
    devset = examples[train_size : train_size + dev_size]
    # Official tutorial uses hover[650:] as a held-out test pool.
    testset = examples[650:]
    return trainset, devset, testset
