from langchain_ollama import ChatOllama
from pathlib import Path
from langchain_core.prompts import PromptTemplate


def main():
    information = Path(__file__).with_name("information.txt").read_text(encoding="utf-8")
    summary_template = """
    Given the information {information} about a person I want you to create:
1. A short summary
2. Two interesting facts about them
"""
    summary_prompt_template = PromptTemplate(
        input_variables={"information"},
        template=summary_template
    )
    llm = ChatOllama(temperature=0, model="gpt-oss:20b-cloud")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
