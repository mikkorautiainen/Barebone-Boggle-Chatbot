"""
A script to run a chatbot using BLOOM-style model.

This script demonstrates how to use the transformers library to
run a chatbot with Hugging Face or local model.
The script is organized into three main sections: configurations,
utility functions, and main execution.
"""

import argparse
import logging
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from typing import Tuple, List

from barebone_response_ranker import ResponseRanker


# ---------------------------
# Section 1: Configurations
# ---------------------------

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = 'TurkuNLP/gpt3-finnish-xl'
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_RETURN_SEQUENCES = 5

ENGINEERED_PROMPT = 'Vastaa tähän kysymykseen sanomalla: '


# ---------------------------
# Section 2: Utility Functions
# ---------------------------

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the BLOOM model and its tokenizer based on the given model name.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded BLOOM model and its tokenizer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Loading model: {model_name} on {device}...')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    return model, tokenizer


def set_logger_level(debug_mode: bool):
    """
    Set the logger level based on the debug mode.

    Args:
        debug_mode (bool): The debug mode setting. If True, set the logger level to DEBUG.
    """
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def normalize_input(user_input: str) -> str:
    """
    Normalize the user input.

    This function may include operations like trimming leading/trailing white spaces,
    converting to lower/upper case, removing special characters, etc. according to the
    specific needs of the project.

    Args:
        user_input (str): The user input to be normalized.

    Returns:
        str: The normalized user input.
    """
    normalized_input = user_input.strip().lower()
    return normalized_input


def validate_input(user_input: str) -> bool:
    """
    Validate the user input.

    This function checks if the user input meets certain conditions, for example,
    if it is not empty, if it is not too long, if it does not contain prohibited
    characters or words, etc.

    Args:
        user_input (str): The user input to be validated.

    Returns:
        bool: True if the input is valid, False otherwise.
    """
    if not user_input:
        return False
    if len(user_input) > 1000:  # Let's assume we don't accept inputs longer than 1000 characters.
        return False
    return True


def filter_incomplete_sentences(text: str) -> str:
    """
    Filter out incomplete sentences from the given text.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    sentences = nltk.sent_tokenize(text)
    complete_sentences = [sentence for sentence in sentences if sentence.endswith(('.', '?', '!'))]
    return ' '.join(complete_sentences)


def generate_response(encoded_input: BatchEncoding, model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer, max_length: int = 512,
                      num_return_sequences: int = 5, top_k: int = 50,
                      temperature: float = 1.0) -> List[str]:
    """
    Generate response sequences using the given model.

    Args:
        encoded_input (BatchEncoding): The tokenized input for the model.
        model (AutoModelForCausalLM): The BLOOM model to generate responses.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained model.
        max_length (int): The maximum length of the generated sequences.
        num_return_sequences (int): The number of sequences to return.
        top_k (int): The number of top K most likely next words to sample from.
        temperature (float): The temperature value for controlling randomness.

    Returns:
        list: A list of generated response sequences.
    """
    attention_mask = encoded_input['attention_mask'].to(model.device)
    input_tokens = encoded_input['input_ids'].to(model.device)

    # Generate responses using top-k sampling
    output_sequences = model.generate(input_tokens,
                                      max_length=max_length,
                                      num_return_sequences=num_return_sequences,
                                      do_sample=True,
                                      no_repeat_ngram_size=2,
                                      pad_token_id=50256,
                                      eos_token_id=50256,
                                      attention_mask=attention_mask,
                                      top_k=top_k,
                                      temperature=temperature)

    # Decode the generated sequences
    responses = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output_sequences]

    return responses


# ---------------------------
# Section 3: Main Function
# ---------------------------

def chatbot_response(question: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, max_length: int,
                     num_return_sequences: int, top_k: int, temperature: float) -> List[str]:
    """
    Generate chatbot responses for a given question.

    Args:
        question (str): The user's question.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained model.
        model (AutoModelForCausalLM): The BLOOM model to generate responses.
        max_length (int): The maximum length of the generated sequences.
        num_return_sequences (int): The number of sequences to return.
        top_k (int): The number of top K most likely next words to sample from.
        temperature (float): The temperature value for controlling randomness.

    Returns:
        list: A list of generated response sequences.
    """
    # Prepend the engineered prompt to the user's question
    input_sequence = ENGINEERED_PROMPT + question

    # Encode the input sequence
    encoded_input = tokenizer.encode_plus(input_sequence, return_tensors='pt')

    # Generate responses
    responses = generate_response(encoded_input, model, tokenizer, max_length, num_return_sequences, top_k, temperature)

    return responses


def run_chatbot(args):
    """
    The main function to run the chatbot.

    Args:
        args (Namespace): The parsed command-line arguments.
    """
    # Initialize logger
    set_logger_level(args.debug)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Initialize the ResponseRanker
    logger.info('Initializing ResponseRanker...')
    response_ranker = ResponseRanker(model, tokenizer, debug=args.debug)

    print(f"\nTervetuloa {args.model_name} chatbottiin! (Kirjoita 'lopeta' poistuaksesi)")

    while True:
        question = input('Sinä: ')
        question = normalize_input(question)
        if question == 'lopeta':
            break
        if not validate_input(question):
            continue

        responses = chatbot_response(question,
                                     tokenizer,
                                     model,
                                     args.max_length,
                                     args.num_return_sequences,
                                     args.top_k,
                                     args.temperature)

        # Filter incomplete sentences from each response
        responses = [filter_incomplete_sentences(response) for response in responses]

        if args.debug:
            for idx, response in enumerate(responses):
                logger.debug(f'\nChatbot {idx + 1}: {response}')

        # Select best response
        response, response_scores = response_ranker.match_question_to_responses(question, responses)

        logger.debug(f'\nResponse_scores: {response_scores}')

        print(f'\nChatbot: {response}')
        print('\n')


def handle_arguments():
    """
    Handle command-line arguments using argparse.

    Returns:
        Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=f'Run a chatbot using the {DEFAULT_MODEL_NAME} model.')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='The name of the pre-trained model to use.')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='The temperature value for controlling randomness.')
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                        help='The number of top K most likely next words to sample from.')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                        help='The maximum length of the generated responses.')
    parser.add_argument('--num_return_sequences', type=int, default=DEFAULT_NUM_RETURN_SEQUENCES,
                        help='The number of sequences to return.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = handle_arguments()
    run_chatbot(args)
