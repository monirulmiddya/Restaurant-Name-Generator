from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = GooglePalm(google_api_key=google_api_key, temperature=0.6)

# name = llm(
#     "I want to open a resturant for indian food . Suggest a fency name for this."
# )
# print(name)

############   SimpleSequentialChain   ################

# prompt_template_name = PromptTemplate(
#     input_variables=["cuisine"],
#     template="I want to open a resturant for {cuisine} food . Suggest a fency name for this.",
# )

# # print(prompt_template_name.format(cuisine="italian"))

# name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
# # name = name_chain.run("Arabic")

# prompt_template_items = PromptTemplate(
#     input_variables=["resturant_name"],
#     template="""Suggest some menu item for {resturant_name}. Return it as a coma separated list""",
# )

# items_chain = LLMChain(llm=llm, prompt=prompt_template_items)

# chain = SimpleSequentialChain(chains=[name_chain, items_chain])

# response = chain.run("Indian")


############   SequentialChain   ################

prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a resturant for {cuisine} food . Suggest a fency name for this.",
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="resturant_name")

prompt_template_items = PromptTemplate(
    input_variables=["resturant_name"],
    template="""Suggest some menu item for {resturant_name}. Return it as a coma separated list""",
)

items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

chain = SequentialChain(
    chains=[name_chain, items_chain],
    input_variables=["cuisine"],
    output_variables=["resturant_name", "menu_items"],
)

response = chain({"cuisine": "Indian"})


print(response)
