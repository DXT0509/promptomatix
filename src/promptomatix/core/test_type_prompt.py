import os
import csv
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
	# Prefer relative import when part of the package
	from .prompts import generate_classification_type_prompt
except Exception:
	# Fallback when running as a script
	from promptomatix.core.prompts import generate_classification_type_prompt

try:
	import litellm
except Exception as _e:
	litellm = None


def _call_openrouter(prompt: str, model_name: str = "openai/gpt-oss-20b") -> str:
	"""Call OpenRouter (LLM) and return content string."""
	if litellm is None:
		raise RuntimeError("litellm is not installed. Please add it to requirements and install.")

	api_key = (
		os.environ.get("OPENROUTER_API_KEY")
		or os.environ.get("API_KEY")
		or os.environ.get("OPENAI_API_KEY")
	)
	if not api_key:
		raise RuntimeError("Missing OPENROUTER_API_KEY in environment.")

	resp = litellm.completion(
		model=model_name,
		messages=[{"role": "user", "content": prompt}],
		api_base="https://openrouter.ai/api/v1",
		api_key=api_key,
		custom_llm_provider="openrouter",
		timeout=15,
		temperature=0.0,
		top_p=0.1,
	)
	return resp.choices[0].message.content


def score_row(pred_answer: str, input_value: str) -> int:
	"""Score per rules: True + input present => 1; False + input missing => 1; else 0."""
	ans = (pred_answer or "").strip().strip('"').strip("'").lower()
	has_input = bool((input_value or "").strip())
	if ans == "true" and has_input:
		return 1
	if ans == "false" and (not has_input):
		return 1
	return 0

BASE_DIR = Path(__file__).resolve().parents[3]
def run(csv_in: str = BASE_DIR / "experiments/dataset/TestClassificationPrompt.csv") -> Path:
	"""
	Read CSV with columns: instruction, input. For each row:
	- Build classification gate prompt via `generate_classification_type_prompt(instruction)`
	- Send to LLM (OpenRouter: openai/gpt-oss-20b)
	- Score per rule
	- Export results to sessions CSV with final average score row.
	Returns the output CSV path.
	"""
	in_path = Path(csv_in)
	if not in_path.exists():
		raise FileNotFoundError(f"Input CSV not found: {in_path}")

	sessions_dir = Path(os.getcwd()) / "sessions"
	sessions_dir.mkdir(parents=True, exist_ok=True)
	ts = int(time.time())
	out_path = sessions_dir / f"test_type_prompt_{ts}.csv"
	# Read input CSV into rows list
	rows: List[Dict] = []
	with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
		reader = csv.DictReader(f)
		for r in reader:
			instruction = r.get("instruction", "")
			input_val = r.get("input", "")
			rows.append({"instruction": instruction, "input": input_val})

	results: List[Dict] = [None] * len(rows)
	total = 0
	dem = 0
	# Concurrency: send up to MAX_CONCURRENT_REQUESTS (default 10) parallel requests
	try:
		max_workers = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
	except Exception:
		max_workers = 10
	max_workers = max(1, min(max_workers, len(rows)))

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_to_idx = {}
		for idx, r in enumerate(rows):
			prompt = generate_classification_type_prompt(r["instruction"])
			future = executor.submit(_call_openrouter, prompt)
			future_to_idx[future] = idx

		for future in as_completed(future_to_idx):
			idx = future_to_idx[future]
			r = rows[idx]
			try:
				resp = future.result()
			except Exception as e:
				resp = f"ERROR: {e}"
			s = score_row(resp, r["input"])
			total += s
			results[idx] = {
				"instruction": r["instruction"],
				"input": r["input"],
				"pred_answer": resp,
				"score": s,
			}
			dem += 1
			print("Running on test: ", dem, "\n")
		
		
	avg = (total / len(results)) if results else 0.0

	# Write output CSV: instruction, input, pred_answer, score; footer average
	fieldnames = ["instruction", "input", "pred_answer", "score"]
	with open(out_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		for row in results:
			writer.writerow(row)
		# footer
		writer.writerow({})
		writer.writerow({"instruction": "AVERAGE", "input": "", "pred_answer": "", "score": avg})

	print(f"Exported results to: {out_path}")
	print(f"Average score: {avg:.4f}")
	return out_path


if __name__ == "__main__":
	# Allow override via env var INPUT_CSV or CLI arg (optional)
	input_csv = os.environ.get("INPUT_CSV") or BASE_DIR / "experiments/dataset/TestClassificationPrompt.csv"
	try:
		run(input_csv)
	except Exception as e:
		print(f"Failed to run: {e}")
