import os
import math
import json
import time
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from typing import List, Optional, Any, Dict
from flashtext import KeywordProcessor
from datasets import load_dataset, Features, Value

# Disable caching to prevent disk overflow when streaming terabyte-scale datasets
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"

KEYWORDS_FILE = "config/keyword-filtering.txt"
OUTPUT_DIR = "data/pretrain_data/raw/redpajama/"
TARGET_SOURCES = ["arxiv", "stackexchange", "c4", "common_crawl"]

class RedpajamaCrawl:
    """Telecom domain filtering pipeline for RedPajama-Data-1T.

    Uses a streaming mechanism to process massive datasets without full local
    downloads, combined with multiprocessing to optimize keyword-based filtering.
    """

    def __init__(
        self,
        keywords_file: str,
        output_dir: str,
        target_sources: List[str],
        batch_size: int = 1000,
        save_interval: int = 2000,
        min_unique_keywords: int = 2,
        min_density_score: float = 0.5,
        num_processes: Optional[int] = None,
    ):
        """Initializes the pipeline with filtering parameters and system resources.

        Args:
            keywords_file: Path to the plain text file containing keywords.
            output_dir: Directory where filtered JSONL files will be stored.
            target_sources: List of RedPajama subsets to process.
            batch_size: Number of samples to accumulate before parallel filtering.
            save_interval: Number of samples to accumulate before saving to disk.
            min_unique_keywords: Minimum number of unique domain terms required.
            min_density_score: Threshold for M/log(N+1) relevance score.
            num_processes: Number of worker processes. Defaults to 50% of CPU cores.
        """
        self.keywords_file = keywords_file
        self.output_dir = output_dir
        self.target_sources = target_sources
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.min_unique_keywords = min_unique_keywords
        self.min_density_score = min_density_score
        self.num_processes = num_processes or max(1, os.cpu_count() // 2)

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _init_worker(keywords_path: str):
        """Initializes KeywordProcessor in each worker process.

        Uses FlashText for O(n) search complexity regardless of keyword count.
        """
        global keyword_processor
        keyword_processor = KeywordProcessor(case_sensitive=True)

        with open(keywords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    keyword_processor.add_keyword(word)

    def _filter_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluates telecom relevance for a single data sample.

        Uses the formula: Score = M / log(N + 1)
        Where M is the number of unique keywords and N is the total word count.

        Returns:
            The enriched item if it passes the threshold, else None.
        """
        text = item.get("text", "")
        if not text:
            return None

        # Calculate word count (N)
        words = text.split()
        n_words = len(words)
        if n_words < 50:
            return None

        # Handle string-encoded meta (common for Common Crawl subset)
        raw_meta = item.get("meta")
        if isinstance(raw_meta, str):
            try:
                item["meta"] = json.loads(raw_meta)
            except json.JSONDecodeError:
                pass

        # Extract keywords and calculate unique count (M)
        found_keywords = keyword_processor.extract_keywords(text)
        unique_keywords = set(found_keywords)
        m_keywords = len(unique_keywords)

        if m_keywords < self.min_unique_keywords:
            return None

        try:
            density_score = m_keywords / math.log(n_words + 1)
        except ValueError:
            return None

        if density_score >= self.min_density_score:
            item["telecom_meta"] = {
                "n_word_count": n_words,
                "m_keyword_count": m_keywords,
                "density_score": round(density_score, 4),
                "matched_terms": list(unique_keywords),
            }
            return item

        return None

    def _save_chunk(
        self, source_name: str, source_dir: str, chunk: List[dict], file_index: int
    ) -> str:
        """Persists a list of filtered samples to a JSONL file."""
        filename = f"{source_name}_{file_index:04d}.jsonl"
        filepath = os.path.join(source_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return filename

    def _get_features(self, source_name: str) -> Features:
        """Defines the explicit dataset schema to prevent type-casting errors."""
        if source_name == "arxiv":
            meta_schema = {
                "arxiv_id": Value("string"),
                "language": Value("string"),
                "timestamp": Value("string"),
                "url": Value("string"),
                "yymm": Value("string"),
            }
        elif source_name == "common_crawl":
            meta_schema = Value("string")
        elif source_name == "c4":
            meta_schema = {
                "url": Value("string"),
                "timestamp": Value("string"),
                "source": Value("string"),
                "language": Value("string"),
            }
        elif source_name == "stackexchange":
            meta_schema = {
                "url": Value("string"),
                "timestamp": Value("string"),
                "source": Value("string"),
                "language": Value("string"),
                "question_score": Value("string"),
            }
        else:
            meta_schema = {"url": Value("string"), "timestamp": Value("string")}

        return Features({
            "text": Value("string"),
            "meta": meta_schema,
            "red_pajama_subset": Value("string"),
        })
    
    def _process_source(self, source_name: str, pool: PoolType):
        """Streams, filters, and saves a specific data subset."""
        print(f"Processing source: {source_name}")

        source_dir = os.path.join(self.output_dir, source_name)
        os.makedirs(source_dir, exist_ok=True)

        try:
            custom_features = self._get_features(source_name)
            dataset = load_dataset(
                "togethercomputer/RedPajama-Data-1T",
                name=source_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
                features=custom_features,
            )
        except Exception as e:
            print(f"Failed to load source {source_name}: {e}")
            return

        buffer_raw = []
        buffer_filtered = []
        file_index = 1
        total_processed = 0
        total_saved = 0
        start_time = time.time()

        try:
            for sample in dataset:
                buffer_raw.append(sample)
                total_processed += 1

                if len(buffer_raw) >= self.batch_size:
                    results = pool.map(self._filter_item, buffer_raw)
                    valid_items = [res for res in results if res is not None]
                    buffer_filtered.extend(valid_items)
                    buffer_raw = []

                    while len(buffer_filtered) >= self.save_interval:
                        chunk = buffer_filtered[: self.save_interval]
                        buffer_filtered = buffer_filtered[self.save_interval :]
                        filename = self._save_chunk(
                            source_name, source_dir, chunk, file_index
                        )
                        total_saved += len(chunk)
                        file_index += 1
                        print(f"Saved: {filename} | Total: {total_saved} | Proc: {total_processed}")

                if total_processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"Processed: {total_processed} | Rate: {rate:.0f} docs/s", end="\r")

            # Final flush for remaining items
            if buffer_raw:
                results = pool.map(self._filter_item, buffer_raw)
                buffer_filtered.extend([res for res in results if res is not None])

            if buffer_filtered:
                self._save_chunk(source_name, source_dir, buffer_filtered, file_index)
        except Exception as e:
            print(f"Stream interrupted for {source_name}: {e}")

    def run(self):
        """Starts the full multi-source filtering pipeline."""
        print(f"Starting pipeline | Workers: {self.num_processes}")

        pool = Pool(
            processes=self.num_processes,
            initializer=self._init_worker,
            initargs=(self.keywords_file,),
        )

        for source_name in self.target_sources:
            for attempt in range(5):
                try:
                    self._process_source(source_name, pool)
                    break
                except Exception as e:
                    print(f"Retry {attempt+1}/5 for {source_name}: {e}")
                    time.sleep(10 * (attempt + 1))

        pool.close()
        pool.join()
        print("Pipeline completed successfully.")

if __name__ == "__main__":
    pipeline = RedpajamaCrawl(
        keywords_file=KEYWORDS_FILE,
        output_dir=OUTPUT_DIR,
        target_sources=TARGET_SOURCES,
        batch_size=200,
        save_interval=500,
        min_unique_keywords=2,
        min_density_score=0.4,
    )
    pipeline.run()