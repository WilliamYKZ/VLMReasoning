from typing import Optional, List, Dict, Tuple
import json
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


class ChartQADataset(Dataset):
    """
    Dataset for ChartQA tasks: reads a JSON file mapping IDs to
    examples, each containing image path, query, answer, and bboxes.
    """
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            data_map: Dict = json.load(f)
        self.examples: List[Dict] = []
        for ex in data_map.values():
            self.examples.append({
                'figure_path':   ex['figure_path'],
                'annotation_path': ex.get('annotation_path'),
                'table_path':      ex.get('table_path'),
                'query':           ex['query'],
                'answer':          ex['answer'],
                'type':            ex.get('type'),
                'figure_bbox':     ex.get('figure_bbox'),
                'x_values':        ex.get('x_values', []),
                'y_values':        ex.get('y_values', []),
                'x_bboxes':        ex.get('x_bboxes', []),
                'y_bboxes':        ex.get('y_bboxes', []),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin]
) -> Tuple[StatefulDataLoader, StatefulDataLoader]:
    # Choose ChartQADataset when using figure_path images
    if config.image_key == 'figure_path':
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

    # Train sampler
    if config.shuffle:
        g = torch.Generator().manual_seed(config.seed)
        train_sampler = RandomSampler(data_source=train_dataset, generator=g)
    else:
        train_sampler = SequentialSampler(data_source=train_dataset)

    train_loader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=train_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # Validation dataset & sampler
    if config.image_key == 'figure_path':
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
    val_sampler = SequentialSampler(data_source=val_dataset)

    val_loader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch,
        sampler=val_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Size of train dataloader: {len(train_loader)}")
    print(f"Size of val dataloader: {len(val_loader)}")
    return train_loader, val_loader
