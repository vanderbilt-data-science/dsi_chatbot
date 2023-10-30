import os
import gradio as gr
import time
import openai
from dotenv import load_dotenv

import pickle
from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# get the api key from the .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# print(openai_api_key)


""" pull the stored embeddings """
embedding_file = hf_hub_download(repo_id="yahanyang777/DSI_web_content", filename="faiss_store_openai.pkl", repo_type="dataset")
with open(embedding_file, 'rb') as f:
  VectorStore = pickle.load(f)

print(VectorStore)

""" initialize all the tools """

template = """
You are the director of DSI and you are cautious about the answer you are giving. You will refuse to answer any questions that may generate an answer that violates the Open AI policy or is not related to DSI.
Use the information provided in the context and the Frequently Asked Questions to answer the question at the end.
• If the question matches one in the Frequently Asked Questions, provide the corresponding answer directly.
• If the question isn't covered in the FAQs but can be inferred from the provided context, use the context to formulate your answer, adhering to the format of the answer in the Frequently Asked Questions section.
• If the question cannot be answered based on either the FAQs or the context, simply state that you don't know. Do not provide inaccurate or made-up information.
Your answers should be:
• Direct and succinct.
• Given in the FAQ format provided below.
• Accompanied by the source website link if available.
• Accurate and directly addressing the user's questions.
{context}
Frequently Asked Questions:
1. Is this an online program?
• The Data Science master’s program at Vanderbilt University is a fully in-person program. Classes are held Monday-Thursdays between the hours of 8:00 am - 5:00 pm.
• https://www.vanderbilt.edu/datascience/academics/msprogram/program-details/
2. Are there any scholarships?
• The Vanderbilt Data Science Institute offers a few full scholarships to our top incoming students each year. To be considered for a scholarship, applicants must apply by the January 15th deadline.
• We also partner with WiTT (Women in Technology in Tennessee) and Vanderbilt ROTC to provide incoming students with 100% tuition scholarships.
• View more details and learn how to apply on our Tuition, Fees, and Scholarships website: https://www.vanderbilt.edu/datascience/academics/msprogram/tuition-fees-and-financials/
3. Can I apply with a 3-year degree?
• Students must have a four-year Bachelor’s Degree (equivalent to a U.S. Bachelor’s degree) to be eligible to apply to the masters in Data Science program.
• https://www.vanderbilt.edu/datascience/academics/msprogram/admissions/
4. Is the GRE required?
• The GRE or GMAT is not required.
• https://www.vanderbilt.edu/datascience/academics/msprogram/admissions/
5. Is there a minimum TOEFL score to apply?
• Proof of English Proficiency is required: If English is not your native language, please submit one of the following:
   o TOEFL scores – Minimum score of 100 total with no sub score of under 21 is required to apply.
   o IELTS scores – Minimum score of 7 total with minimum sub-scores of 6.5 required to apply.
   o Duolingo test with sub-scores – Minimum score of 130 required to apply.
   o Evidence that you have a degree from an English-speaking institution of higher education.
• https://www.vanderbilt.edu/datascience/academics/msprogram/admissions/
6. Do you offer application fee waivers?
• There is no application fee; it is free to apply!
• https://www.vanderbilt.edu/datascience/academics/msprogram/admissions/
7. How long is the program?
• The Vanderbilt Master of Science in Data Science is an in-person 2-year, 16-course (48 credits) program.
• https://www.vanderbilt.edu/datascience/academics/msprogram/curriculum/
8. What time are classes held?
• The Data Science master’s program at Vanderbilt University is a fully in-person program. Classes are held Monday-Thursdays between the hours of 8:00 am - 5:00 pm.
• https://www.vanderbilt.edu/datascience/academics/msprogram/program-details/
Question: {question}
Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template,)

chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

answer_chain = RetrievalQA.from_chain_type(
    chat_model,
    retriever=VectorStore.as_retriever(),
    memory=memory,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

suggested_ques_prompt = PromptTemplate(
            input_variables=["chat_history", "answer"],
            template="Based on the recent chat history:{chat_history} and especially the answer: {answer}, suggest two relevant and short follow-up questions.",
        )
suggested_ans_chain = LLMChain(llm=chat_model, prompt=suggested_ques_prompt)


def generate_follow_ups(answer, chat_history):
    prompt = PromptTemplate(
            input_variables=["chat_history", "answer"],
            template="Based on the recent chat history:{chat_history} and especially the answer: {answer}, suggest two relevant and short follow-up questions.",
        )
    chain = LLMChain(llm=chat_model, prompt=prompt)

    follow_ups = chain.run({"answer":answer, "chat_history":chat_history})
    
    return follow_ups


def respond(usr_message, chat_history=None):
    try:
        if chat_history is None:
            chat_history = []
            
        # Attempt to get a response from the OpenAI model
        bot_message = answer_chain({"query": usr_message}, return_only_outputs=True)['result']
        
        # Generate follow-up questions
        follow_ups = suggested_ans_chain.run({"answer":bot_message, "chat_history":chat_history})
        
        bot_message += "\n\nBelow are some suggested follow-up questions:\n"
        bot_message += follow_ups
            
        # Store the user message and bot response in the database
        chat_history.append((usr_message, bot_message))
        
        time.sleep(1)
        return "", chat_history

    except openai.error.OpenAIError as e:
        # Handle OpenAI-specific errors
        error_message = f"OpenAI API Error: {e}"
        print(error_message)
        return error_message, chat_history

    except Exception as e:
        # Handle other unexpected errors
        error_message = f"Unexpected error: {e}"
        print(error_message)
        return error_message, chat_history





def slow_echo(usr_message, chat_history):
    try:
        # Attempt to get a response from the OpenAI model
        bot_message = answer_chain({"query": usr_message}, return_only_outputs=True)['result']
        
        # Generate follow-up questions
        follow_ups = generate_follow_ups(bot_message, chat_history)
        
        bot_message += "\n\nBelow are some suggested follow-up questions:\n"
        bot_message += follow_ups
            
        # Store the user message and bot response in the database
        time.sleep(1)
        
        yield bot_message

    except openai.error.OpenAIError as e:
        # Handle OpenAI-specific errors
        error_message = f"OpenAI API Error: {e}"
        print(error_message)

    except Exception as e:
        # Handle other unexpected errors
        error_message = f"Unexpected error: {e}"
        print(error_message)
        

demo = gr.ChatInterface(
    slow_echo,
    chatbot=gr.Chatbot(height=300, label="DSI Web Chatbot"),
    textbox=gr.Textbox(label='Type in / choose your questions about DSI here and press Enter!',
                       placeholder='Type in your questions.', container=False, scale=8),
    # title="DSI Web - LangChain Bot",
    examples=["Is this an online program?", "Are there any scholarships?", "Is the GRE required?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
).queue()
