import argparse
import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv
from train_dspy_optimizer import CoT

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

from sentence_transformers import SentenceTransformer, util


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# Initialize embedding model once (to avoid redundant loading)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def our_metric(gold: str, pred: str, trace=None) -> float:
    """
    Calculates answer similarity using cosine similarity.

    Parameters:
    - gold (str): The expert-provided correct answer.
    - pred (str): The LLM-generated response.
    - trace (optional): Not used, but kept for compatibility.

    Returns:
    - float: Similarity score (0 to 1), where 1 means identical and 0 means completely different.
    """
    try:
        # Compute embeddings
        gold_embedding = embedding_model.encode(gold.answer, convert_to_tensor=True)
        pred_embedding = embedding_model.encode(pred.answer, convert_to_tensor=True)

        # Compute cosine similarity
        similarity_score = util.cos_sim(gold_embedding, pred_embedding).item()

        return float(similarity_score)  # Score between 0 and 1

    except Exception as e:
        print(f"Error in our_metric: {e}")
        return 0.0  # Default to 0 if error occurs


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

    # # Evaluate optimized program
    # print("Evaluate optimized program...")
    # evaluate(zeroshot_optimized_program, devset=devset[:])
    
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_PATH = os.path.join(ROOT_DIR, "mipro_zeroshot_optimized_v2.json")

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
    prompt["question"] = prompt["question"] + \
        f"The experts told me that {result.answer}. This was based on the reasoning: {result.reasoning} and the context provided." + \
        "\nPlease recommend a stock in the following format:\n[Stock Recommendation]: <Stock Ticker>\n[Justification]: <Why this stock is a good choice>."
    print(json.dumps(prompt, indent=4))


def mock_dspy_perdict():
    args = parse_args()
    prompt: dict = args.prompt

    prompt["reasoning"] = "you should look at the news to answer this question"

    print(json.dumps(prompt, indent=4))

if __name__ == "__main__":
    OPTIMIZER_PATH = "mipro_zeroshot_optimized_v1.json"
    data_path = "modules/q_and_a_dataset_generator/data/filtered_training_data_based_on_stock_metric.json"
    train_dspy_optimizer(data_path)
    
    # test_cases = [
    # ("Tesla stock is a strong buy due to increasing EV demand and strong financials.",
    #  "Tesla is a good investment because of rising demand for electric cars and strong revenue."),
    
    # ("Investing in Apple is wise due to its strong earnings and innovative products.",
    #  "AAPL is a great long-term investment given its steady growth and leadership in technology."),
    
    # ("Bitcoin is highly volatile, making it a risky but potentially high-reward investment.",
    #  "Bitcoin is a safe asset with no volatility, making it risk-free."),
    
    # ("The S&P 500 is a good benchmark for diversified stock investments.",
    #  "The Federal Reserve sets interest rates to control inflation."),
    # ]
    
    # for i, (gold, pred) in enumerate(test_cases):
    #     print(f"Test Case {i+1}:")
    #     print(f"Gold Answer: {gold}")
    #     print(f"Predicted Answer: {pred}")
    #     print(f"Cosine Similarity: {our_metric(gold, pred):.2f}")
    
    
    # dspy_perdict()
