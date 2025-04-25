from typing import Optional, List, Dict
import json
import os
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


class ChartQADataset(Dataset):
    """
    Dataset for ChartQA tasks: reads a JSON file mapping IDs to examples,
    each containing image path, query, answer, and bounding-box info.
    """
    def __init__(self, json_path: str):
        # Load the full mapping
        with open(json_path, 'r') as f:
            data_map: Dict = json.load(f)
        # Convert to list of examples
        self.examples: List[Dict] = []
        for ex_id, ex in data_map.items():
            example = {
                'figure_path': ex['figure_path'],
                'annotation_path': ex.get('annotation_path'),
                'table_path': ex.get('table_path'),
                'query': ex['query'],
                'answer': ex['answer'],
                'type': ex.get('type'),
                'figure_bbox': ex.get('figure_bbox'),
                'x_values': ex.get('x_values', []),
                'y_values': ex.get('y_values', []),
                'x_bboxes': ex.get('x_bboxes', []),
                'y_bboxes': ex.get('y_bboxes', []),
            }
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        # Return one example dict
        return self.examples[idx]


def create_dataloader(config: DataConfig,
                      tokenizer: PreTrainedTokenizer,
                      processor: Optional[ProcessorMixin]) -> None:
    # Determine which dataset to use
    if getattr(config, 'task', '').lower().startswith('chartqa'):
        train_dataset = ChartQADataset(config.train_files)
    else:
        train_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            max_prompt_length=config.max_prompt_length,
            truncation='right',
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

    # Sampler setup
    if config.shuffle:
        g = torch.Generator()
        g.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=g)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # Validation split uses same logic
    if getattr(config, 'task', '').lower().startswith('chartqa'):
        val_dataset = ChartQADataset(config.val_files)
    else:
        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            max_prompt_length=config.max_prompt_length,
            truncation='right',
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )
    val_batch = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
