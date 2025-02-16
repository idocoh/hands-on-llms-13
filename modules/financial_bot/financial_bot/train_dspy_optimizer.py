import json
import random
from pathlib import Path
import dspy
from dspy.teleprompt import MIPROv2
from sentence_transformers import SentenceTransformer, util

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_answer_similarity(response: str, expert_answer: str, _: str=None) -> float:
    """
    Measures how similar the generated response is to a reference expert answer.

    Parameters:
    - response (str): The model-generated answer.
    - expert_answer (str): The expert-provided correct answer.

    Returns:
    - Similarity score (0-1): Higher means the response is closer to expert analysis.
    """
    response = response["answer"]
    expert_answer = expert_answer["answer"]
    response_embedding = embedding_model.encode(response)
    expert_embedding = embedding_model.encode(expert_answer)

    # Compute cosine similarity
    similarity_score = util.cos_sim(response_embedding, expert_embedding).item()

    return similarity_score


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
    
    
def optimize_prompt(program, trainset):
    # Initialize optimizer
    teleprompter = MIPROv2(
        metric=calculate_answer_similarity,
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
    
    return zeroshot_optimized_program


def train_dspy_optimizer(data_path, optimizer_path):
    with open(data_path, 'r') as f:
        json_data = json.load(f)

    trainset = []

    for example in json_data:
        question = example["about_me"].split("\n")[-1]
        gold_reasoning = example["context"]
        answer = example["response"]

        trainset.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

    trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]

    program = CoT()
    zeroshot_optimized_program = optimize_prompt(program, trainset)
    zeroshot_optimized_program.save(optimizer_path)


if __name__ == "__main__":
    OPTIMIZER_PATH = str(Path(__file__).parent.parent.parent.parent / "modules/financial_bot/financial_bot/mipro_zeroshot_optimized_v1.json")
    DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "modules/q_and_a_dataset_generator/data/filtered_training_data_based_on_stock_metric.json")
    train_dspy_optimizer(DATA_PATH, OPTIMIZER_PATH)