import argparse
import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv
from train_dspy_optimizer import CoT

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_PATH = os.path.join(ROOT_DIR, "mipro_zeroshot_optimized_v0.json")

PROMPT_KEYS = ["about_me", "context", "question"]

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def validate_json(json_str: str) -> dict:
    try:
        prompt = json.loads(json_str)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON string.")

    if not all(key in prompt for key in PROMPT_KEYS):
        raise argparse.ArgumentTypeError(f"Prompt must contain the following keys: {PROMPT_KEYS}")

    return prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize prompt with MIPROv2 optimizer.")
    parser.add_argument("--prompt", type=validate_json, required=True, 
                        help="The prompt to optimize.")
    return parser.parse_args()

def dspy_perdict():
    args = parse_args()
    prompt: dict = args.prompt
    
    if not Path(OPTIMIZER_PATH).exists():
        raise FileNotFoundError(f"Optimizer not found at {OPTIMIZER_PATH}")

    cot = CoT()
    cot.load(OPTIMIZER_PATH)

    question = prompt["about_me"] + " " + prompt["context"] + " " + prompt["question"]
    result = cot(question=question)

    prompt["reasoning"] = result.reasoning
    prompt["dspy_answer"] = result.answer
    prompt["question"] = prompt["question"] + f"\nYou can use the following expert answer as a reference {result.answer}, given the experts resoning is: {result.reasoning}" + "\nRecommend a stock in the following format:\n[Stock Recommendation]: <Stock Ticker>\n[Justification]: <Why this stock is a good choice>. Make sure that the recommendation is based on the context provided"

    print(json.dumps(prompt, indent=4))


def mock_dspy_perdict():
    args = parse_args()
    prompt: dict = args.prompt

    prompt["reasoning"] = "you should look at the news to answer this question"

    print(json.dumps(prompt, indent=4))

if __name__ == "__main__":
    dspy_perdict()
