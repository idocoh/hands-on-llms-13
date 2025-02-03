# Semi-automatic Q&A dataset generation

## Context

Fine-tuning is about adjusting the model weights to maximize performance on a narrowly defined specific task, for example, provide the best possible financial advice.

In a real-world project, we would hire a team of financial experts, to bootstrap an initial dataset of pairs (question, answer). In this tutorial, we will follow a semi-automatic approach, and use a general LLM, like ChatGPT, to bootstrap a reasonable training set.

This dataset should resemble as much as possible the actual questions, and answers we expect, from this model once deployed. This is the dataset we will use to **fine-tune** our LLM.


## Quick set up

* Set up virtual env using Poetry (if you get an error message that poetry.lock is out of date, run `poetry lock --no-update` and then re-run):
    ```
    $ make init
    ```

* Run the init script for environment variables
    ```
    $ cp set_env_variables_template.sh set_env_variables.sh
    ```

* Edit set_env_variables.sh to include your external service credentials.
    
* Run the init script for environment variables
    ```
    $ . ./set_env_variables.sh
    ```

* Generate a sample of training data
    ```
    $ make training-data
    ```

* Generate sample data for DSPY optimizer
    ```
    $ make data-w-stocks
    ```
**Note:** If not working due to the error message: "libcudnn.so.8: cannot open shared object file: No such file or directory", try to copy all the "nvidia" prefixed libraries from the training pipeline poetry folder (in the /lib/python3.10/site-packages/ sub-folder), to the same subfolder in this module poetry folder (in the exact same subfolder). The poetry folder can be found by running from the referenced pipeline's folder the following command:
    ```
    $ poetry env list --full-path
    ``` 

## Not used here but might be useful later on

Unused pieces of code that can be useful later on, for example, to backfill the feature store
or the vector db.

* Get around `18k` news from January 2023 from Alpaca into a JSON file:
    ```
    $ make download
    ```

* Push this JSON file to Qdrant DB as embeddings
    ```
    $ make embed
    ```
**Note:** If not working due to the error message: "libcudnn.so.8: cannot open shared object file: No such file or directory", try to copy all the "nvidia" prefixed libraries from the training pipeline poetry folder, which can be found by running from the trining pipeline folder the following command:
    ```
    $ poetry env list --full-path
    ``` 

## References
> **A bit more about prompt engineering**
> Here is a recent prompt engineering idea we can use with ChatGP
> https://twitter.com/jeremyphoward/status/1689464587077509120
