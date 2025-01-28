import json
import logging

import fire
from datasets import Dataset

from financial_bot.evaluate_stocks import get_stock_metrics
from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (  # context_entity_recall,; context_relevancy,; context_utilization,
        answer_similarity,
        context_recall,
        faithfulness,
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
        #context_utilization,
        context_relevancy,
        context_recall,
        answer_similarity,
        #context_entity_recall,
        #answer_correctness,
        faithfulness
    ]

    total_scores = {
        'ragas_score': 0.0,
        'context_relevancy': 0.0,
        'context_recall': 0.0,
        'answer_similarity': 0.0,
        'faithfulness': 0.0
    }
    count = 0

    results = []
    rates = []
    did_outperforms = []
    date_start = "2024-01-21"
    date_end = "2024-01-28"
    with open(testset_path, "r") as f:
        data = json.load(f)
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            
            rate, did_outperform = get_stock_metrics(response, date_start, date_end, verbose=False)
            rates.append(rate)
            did_outperforms.append(did_outperform)
            
            logger.info("Score=%s", evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"], metrics=metrics))

            logger.info(f"Mean rate: {sum(rates)/len(rates)}")
            logger.info(f"Mean outperforms: {sum(did_outperforms)/len(did_outperforms)}")
    
            score = evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"], metrics=metrics)
            logger.info("Score=%s", score)

            # Accumulate scores
            for key in total_scores:
                total_scores[key] += score.get(key, 0.0)
            count += 1

            # Store input_payload, response, and scores
            results.append({
                "input_payload": input_payload,
                "context": output_context,
                "GT": elem["response"],
                "response": response,
                "scores": score,
                "rate": rate,
                "did_outperform": did_outperform
            })

    # Calculate averages
    average_scores = {key: total / count for key, total in total_scores.items()}
    logger.info("Average Scores=%s", average_scores)

    # Write results to a JSON file
    with open("results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    return response


if __name__ == "__main__":
    fire.Fire(run_local)