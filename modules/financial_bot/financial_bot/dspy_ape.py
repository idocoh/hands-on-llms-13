import argparse
import json
import random
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

OPTIMIZER_PATH = "mipro_zeroshot_optimized_v0.json"

# Mock implementation of our_metric
class MockMetric:
    def __init__(self):
        pass

    def evaluate(self, program_output, reference_output):
        # Simulate evaluation logic with a random score
        return random.uniform(0, 1)  # Return a random score between 0 and 1

# Use the mock metric in place of the actual metric
our_metric = MockMetric()

# Define or import evaluate function
def evaluate(program, devset):
    # Implement the evaluation logic here
    pass

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)
    
def optimize_prompt(program, trainset, devset):
    # Initialize optimizer
    teleprompter = MIPROv2(
        metric=our_metric,
        auto="light",  # Can choose between light, medium, and heavy optimization runs
    )

    # Optimize program
    print("Optimizing zero-shot program with MIPRO...")
    zeroshot_optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=0, # ZERO FEW-SHOT EXAMPLES
        max_labeled_demos=0, # ZERO FEW-SHOT EXAMPLES
        requires_permission_to_run=False,
    )

    # Evaluate optimized program
    print("Evaluate optimized program...")
    evaluate(zeroshot_optimized_program, devset=devset[:])
    
    return zeroshot_optimized_program

def train_dspy_optimizer(data_path):
    # Load data from JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Shuffle data to ensure randomness
    random.shuffle(data)

    # Split data into 80% training and 20% evaluation
    split_index = int(0.8 * len(data))
    trainset = data[:split_index]
    devset = data[split_index:]

    program = CoT()
    zeroshot_optimized_program = optimize_prompt(program, trainset, devset)
    zeroshot_optimized_program.save(OPTIMIZER_PATH)


PROMPT_KEYS = ["about_me", "context", "question"]

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

def main():
    args = parse_args()
    prompt: dict = args.prompt
    
    if not Path(OPTIMIZER_PATH).exists():
        raise FileNotFoundError(f"Optimizer not found at {OPTIMIZER_PATH}")


    cot = dspy.ChainOfThought("question -> answer")
    cot.load(OPTIMIZER_PATH)

    # Optimize the prompt
    question = prompt["about_me"] + " " + prompt["context"] + " " + prompt["question"]
    result = cot(question=question)
    print(result)

    prompt["reasoning"] = f"You can use the following expert answer as a reference: {result.answer}"

    print(json.dumps(prompt, indent=4))

if __name__ == "__main__":
    data_path = "modules/q_and_a_dataset_generator/data/filtered_training_data_based_on_stock_metric.json"
    train_dspy_optimizer(data_path)
