Agent Framework Test Case

This project showcases an intelligent agent workflow designed to generate questions and answers on a specific topic. The system incorporates three key agents, each with a distinct role.

Agents

1. Researcher Agent
-   Responsibility: Develop ideas for teaching someone new to the subject.
-   Implementation: Tokenize the input content and generate ideas for teaching.

2. Writer Agent
- Responsibility: Use the Researcher’s ideas to write a piece of text to explain the topic.
- Implementation: Utilize a language model (e.g., BART) to generate teaching text based on the provided ideas.

3. Examiner Agent
- Responsibility: Craft 2-3 test questions to evaluate understanding of the created text, along with the correct answers.
- Implementation: Create test questions based on the generated teaching text.
   

Requirements

- nltk==3.6.5
- torch==1.10.0
- transformers==4.14.3

License

This project is licensed under the [MIT License](LICENSE).

Acknowledgments

- CrewAI framework: [https://github.com/joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI)
- Hugging Face Transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
