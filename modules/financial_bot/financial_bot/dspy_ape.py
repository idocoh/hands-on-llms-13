import argparse
import json

# import dotenv

# dotenv.load_dotenv()

# PATH_TO_OPTIMIZER = Path(_file_).parent / "optimizer" / "mipro_optimizer_v2_heavy.json"

# if not PATH_TO_OPTIMIZER.exists():
#     raise FileNotFoundError(f"Optimizer not found at {PATH_TO_OPTIMIZER}")

PROMPT_KEYS = ["about_me", "old_context", "question"]

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

    # # Configure the LM in DSPy
    # lm = dspy.LM("openai/gpt-4o-mini")
    # dspy.configure(lm=lm)

    # cot = dspy.ChainOfThought("question -> answer")
    # cot.load(PATH_TO_OPTIMIZER)

   
    # Optimize the prompt
    # question = prompt["about_me"] + " " + prompt["old_context"] + " " + prompt["question"]
    # result = cot(question=question)

    prompt["reasoning"] = "you should look at the news to answer this question"
    # prompt["question"] = prompt["question"] + "\n" + f"You can use the following expert answer as a reference: {result.answer}"

    print(json.dumps(prompt, indent=4))

if __name__ == "__main__":
    main()