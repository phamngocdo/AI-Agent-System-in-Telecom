import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable

from datasets import load_dataset, Dataset
from src.utils.logger import *

class DatasetLoader(ABC):
    def __init__(self, tokenizer, repo_id: str, train_file: str = "train.json", test_file: str = "test.json", data_dir: str = "data"):
        """
        Args:
            repo_id: Hugging Face dataset repo
            train_file: Name of train file
            test_file: Name of test file
        """
        super().__init__()
        self.tokenizer = tokenizer
        self._dataset_name = repo_id.replace("/", "_")
        self.repo_id = repo_id
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, name: str):
        self._dataset_name = name

    @abstractmethod
    def formatting_train_prompts_func(examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Convert a batch of raw samples into model prompts.
        Must return a dict of lists (HF map-compatible).
        Example return: {"text": [prompt1, prompt2, ...]}
        """
        pass

    @abstractmethod
    def formatting_test_prompts_func(examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Convert a batch of raw samples into model prompts.
        Must return a dict of lists (HF map-compatible).
        Example return: {"text": [prompt1, prompt2, ...]}
        """
        pass

    def load(self, splits: Iterable[str] = ("train", "test")) -> Dict[str, Dataset]:
        """
        Load dataset splits.
        - Always cache raw first
        - Formatting is applied on top of raw cache

        Args:
            splits: ("train",), ("test",) or ("train", "test")

        Returns:
            Dict[str, Dataset]
        """

        splits = set(splits)
        assert splits.issubset({"train", "test"}), "splits must be train/test"

        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        raw_dir = os.path.join(dataset_dir, "raw")
        fmt_dir = os.path.join(dataset_dir, "formatted")

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(fmt_dir, exist_ok=True)

        result = {}

        raw_data = {}

        for split in splits:
            raw_path = os.path.join(raw_dir, f"{split}.json")

            if os.path.exists(raw_path):
                log_info(f"Loading raw {split} from local: {raw_path}")
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw_data[split] = Dataset.from_list(json.load(f))

        missing = splits - raw_data.keys()
        if missing:
            log_info(f"Loading raw splits from HF: {missing}")

            data_files = {}
            if "train" in missing:
                data_files["train"] = self.train_file
            if "test" in missing:
                data_files["test"] = self.test_file

            hf_ds = load_dataset(self.repo_id, data_files=data_files)

            for split in missing:
                ds = hf_ds[split]
                raw_data[split] = ds

                self._save_split(ds, split, raw_dir)

        for split, ds in raw_data.items():
            fmt_path = os.path.join(fmt_dir, f"{split}.json")

            if os.path.exists(fmt_path):
                log_info(f"Loading formatted {split} from cache: {fmt_path}")
                with open(fmt_path, "r", encoding="utf-8") as f:
                    result[split] = Dataset.from_list(json.load(f))
                continue

            log_info(f"Formatting {split}")
            if split == "train":
                fmt_ds = ds.map(
                    self.formatting_train_prompts_func,
                    batched=True,
                    remove_columns=ds.column_names,
                )
            else:
                fmt_ds = ds.map(
                    self.formatting_test_prompts_func,
                    batched=True,
                    remove_columns=ds.column_names,
                )

            result[split] = fmt_ds
            self._save_split(fmt_ds, split, fmt_dir)

        for k, v in result.items():
            log_info(f"{k}: {len(v)} samples")

        return result

    def _save_split(self, dataset: Dataset, split: str, out_dir: str):
        path = os.path.join(out_dir, f"{split}.json")
        log_info(f"Caching {split} → {path}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError(
            "DatasetLoader is for loading/saving only, not iteration."
        )
