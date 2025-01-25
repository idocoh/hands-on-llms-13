import argparse
import json
import random
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

OPTIMIZER_PATH = "modules/financial_bot/financial_bot/mipro_zeroshot_optimized_v0.json"

a = 0.01

def our_metric(gold, pred, trace=None):
    global a
    a += 0.1
    # Simulate evaluation logic with a random score
    return a


# Define or import evaluate function
def evaluate(program, devset):
    # Implement the evaluation logic here
    pass

class CoT(dspy.Module):
    def _init_(self):
        super()._init_()
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
        json_data = json.load(f)

    # convert data to a list
    data = []

    
    # official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
    for example in json_data:
        # question = example["about_me"].split("\n")[-1]  # Extracts the question
        question = example["about_me"] 
        gold_reasoning = example["context"]  # Context is treated as the reasoning
        answer = example["response"]  # The response is treated as the answer

        data.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
    

    # Shuffle data to ensure randomness
    random.shuffle(data)

    # Split data into 80% training and 20% evaluation
    split_index = int(0.8 * len(data))
    trainset = data[:split_index]
    devset = data[split_index:]

    trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]
    devset = [dspy.Example(**x).with_inputs("question") for x in devset]
        
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

    cot = CoT()
    cot.load(OPTIMIZER_PATH)

    # Optimize the prompt
    question = prompt["about_me"] + " " + prompt["context"] + " " + prompt["question"]
    result = cot(question=question)
    # print(result)

    # prompt["reasoning"] = "you should look at the news to answer this question"
    prompt["question"] = prompt["question"] + "\n" + f"You can use the following expert answer as a reference: {result.answer}"

    print(json.dumps(prompt, indent=4))

if __name__ == "_main_":
    main()
    # data_path = "modules/q_and_a_dataset_generator/data/filtered_training_data_based_on_stock_metric.json"
    # train_dspy_optimizer(data_path)