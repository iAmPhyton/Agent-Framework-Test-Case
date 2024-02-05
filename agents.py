import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
import transformers

if not transformers.file_utils.is_offline_mode() and not transformers.file_utils.is_datasets_available():
    transformers.file_utils.get_from_cache("facebook/bart-large-cnn")

nltk.download('punkt')

class ResearcherAgent:
    def process_message(self, message):
        if message.get("content"):
            ideas = self.generate_ideas(message["content"])
            message["next_agent"] = 'writer'
            message["next_content"] = ideas
        return message

    def generate_ideas(self, content):
        sentences = sent_tokenize(content)
        words = [word_tokenize(sentence) for sentence in sentences]
        return f"Ideas for teaching: {sentences} / {words}"


class WriterAgent:
    def __init__(self):
        # Using a zero-shot classification pipeline with a publicly accessible model
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-cnn")

    def process_message(self, message):
        if message.get("content"):
            text = self.write_text(message["content"])
            message["next_agent"] = 'examiner'
            message["next_content"] = text
        return message

    def write_text(self, ideas):
        # Using a zero-shot classification to generate teaching text
        prompt = "Teach about the following ideas: " + ideas
        generated_text = self.classifier(prompt, candidate_labels=["Artificial Intelligence", "Machine Learning", "Cryptocurrency"])
        return f"Teaching Text:\n\n{generated_text['sequences'][0]['sequence']}"

class ExaminerAgent:
    def process_message(self, message):
        if message.get("content"):
            questions = self.craft_questions(message["content"])
            print("Test Questions:")
            for question, answer in questions.items():
                print(f"Q: {question}\nA: {answer}\n")
        return message

    def craft_questions(self, text):
        questions = {
            "What is the main idea of the machine learning?": "Machine learning refers to the general use of algorithms and data to create autonomous or semi-autonomous machines",
            "What is the main idea of the artificial intelligence?": "AI is a field, which combines computer science and robust datasets, to enable problem-solving.",
            "Provide an example of a ML model": "k Nearest Neighbor (kNN)"
        }
        return questions


if __name__ == "__main__":
    researcher_agent = ResearcherAgent()
    writer_agent = WriterAgent()
    examiner_agent = ExaminerAgent()

    # Initial message to start the project
    message = {"Agent Framework Test Case": "Generating questions and answers on specific topics"}

    # Running the agents in sequence
    message = researcher_agent.process_message(message)
    message = writer_agent.process_message(message)
    message = examiner_agent.process_message(message)
