# Barebone-Boogie-Chatbot
  
Welcome to the Barebone-Boogie-Chatbot project!  
This project utilizes the power of the transformers and BLOOM-style models to create a local chatbot.  
The chatbot is designed as a developmental tool for machine learning enthusiasts and researchers.

For a polished comprehensive local chatbot solution, consider [Freedom GPT](https://github.com/ohmplatform/FreedomGPT) instead.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Barebone-Boogie-Chatbot is an efficient compact command-line chatbot designed to evaluate and test different machine learning models.

The Barebone-Boogie-Chatbot is a Python-based chatbot project that uses the transformers library and the AutoModelForCausalLM model. It has the capability to run a chatbot with a Hugging Face model or a local custom model.

This project serves as a resourceful tool to explore the possibilities of AI in the field of Natural Language Processing (NLP).  
This project is designed in such a way that machine learning enthusiasts and researchers can tweak and adjust the configurations to effectively assess their new models.  

This minimal local chatbot solution  can serve as a foundation for more complex projects.  

### Key Features
- Model Flexibility: The software allows users to input different pre-trained models by name. This can be a local model or models from the Hugging Face library.
- Adjustable Parameters: It offers control over several parameters such as temperature, top_k, maximum response length, and the number of return sequences to enhance the precision and customization of the model’s outputs.
- Hardware Compatibility: While it's designed to run optimally on a GPU for faster computation, it also works with CPUs, ensuring functionality across a variety of hardware setups.

## Getting Started

To get a local copy up and running, follow these simple steps:
1. Clone the repo
```bash
git clone https://github.com/mikkorautiainen/Barebone-Boogie-Chatbot.git
```
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage

The chatbot is composed of two main Python scripts:
- `barebone_chatbot.py`: This is the main script that handles user interaction and utilizes the model to generate responses to user's input.
- `barebone_response_ranker.py`: This script contains a class ResponseRanker that ranks the generated responses based on their suitability to the user's question.

To start using the Barebone-Boogie-Chatbot, navigate to the project directory and run the following command:
```bash
python3 barebone_chatbot.py
```

### Options
- `-h, --help`: Displays the help message and exits the program.
- `--model_name MODEL_NAME`: Specifies the pre-trained model to use.
- `--temperature TEMPERATURE`: Controls the randomness of the model’s outputs.
- `--top_k TOP_K`: Determines the number of most likely next words the model samples from.
- `--max_length MAX_LENGTH`: Sets the maximum length of the generated responses.
- `--num_return_sequences NUM_RETURN_SEQUENCES`: Defines the number of sequences to return.
- `--debug`: Enable debug mode.

The chatbot is highly customizable. You can adjust parameters like temperature, the number of top K most likely next words to sample from, the maximum length of the generated sequences and the number of sequences to return.

### Example Usage

For instance, to run the chatbot with a temperature of 0.7, top_k of 500, and a maximum response length of 50, the command would be:
```bash
python3 barebone_chatbot.py --temperature 0.7 --top_k 500 --max_length 50
```

To exit the chatbot, enter 'lopeta' as your input.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create.  
Any contributions you make are greatly appreciated !!!
1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License

Distributed under the MIT License. See ´LICENSE´ for more information.

## Acknowledgements
- [CRM-service Oy](https://crmservice.com)
- [Hugging Face's Transformers](https://github.com/huggingface/transformers)
- [NLTK](https://www.nltk.org)
- [PyTorch](https://pytorch.org)
- [TurkuNLP](https://turkunlp.org)
