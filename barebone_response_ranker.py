"""
ResponseRanker Module

This module contains the ResponseRanker class which ranks a list of responses
to a question using a pretrained BLOOM model.
The ranking is based on the suitability of the response to the given question.

Example:
    For testing module, run the script with a Python interpreter:
    $ python3 barebone_response_ranker.py
"""

import logging
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple


class ResponseRanker:
    """
    ResponseRanker ranks a list of responses to a question using a pretrained BLOOM model.
    """

    def __init__(self, model, tokenizer, min_expected_score: float = -15, debug: bool = True):
        """
        Initialize the ResponseRanker with the specified BLOOM model and tokenizer.

        Args:
            model (AutoModelForCausalLM): The pretrained BLOOM model to use.
            tokenizer (AutoTokenizer): The tokenizer corresponding to the BLOOM model.
            min_expected_score (float): The minimum expected score for a response.
            debug (bool): Whether to print debug information.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.min_expected_score = min_expected_score
        self.debug = debug

        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    def _compute_response_score(self, input_text: str, response: str) -> float:
        """
        Compute the score for a response to the input text using the BLOOM model.

        Args:
            input_text (str): The input text.
            response (str): The response to score.

        Returns:
            float: The score of the response.
        """
        combined_text = f'{input_text} {response}'
        input_ids = self.tokenizer.encode(combined_text, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            output = self.model(input_ids, labels=input_ids)
            logits = output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return losses.mean().item()

    def match_question_to_responses(self, question: str, responses: List[str]) -> Tuple[str, List[float]]:
        """
        Match a question to the best response from a list of responses.

        Args:
            question (str): The question to match.
            responses (list[str]): The list of responses to rank.

        Returns:
            tuple[str, list[float]]: The best response and a list of scores for each response.
        """
        best_score = float('inf')
        best_response = None
        best_response_scores = None

        for response in responses:
            response_sentences = sent_tokenize(response)
            sentence_scores = []

            for sentence in response_sentences:
                sentence_score = self._compute_response_score(question, sentence)
                sentence_scores.append(sentence_score)

                if sentence_score < self.min_expected_score:
                    break

            if len(sentence_scores):
                response_score = sum(sentence_scores) / len(sentence_scores)
            else:
                response_score = 0

            if response_score < best_score:
                best_score = response_score
                best_response = response
                best_response_scores = sentence_scores

        return best_response, best_response_scores


# For demo standalone use
if __name__ == '__main__':
    model_name = 'TurkuNLP/gpt3-finnish-xl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ranker = ResponseRanker(model, tokenizer)
    input_text = 'Mikä on Suomen pääkaupunki?'
    candidate_responses = ['Helsinki on Suomen pääkaupunki. Se sijaitsee Etelä-Suomen rannikolla.',
                           'Suomi on maa Pohjois-Euroopassa. Se rajautuu Ruotsiin, Norjaan ja Venäjään.']
    best_response, best_response_scores = ranker.match_question_to_responses(input_text, candidate_responses)
    print('Best response:', best_response)
    print('Scores per sentence:', best_response_scores)
