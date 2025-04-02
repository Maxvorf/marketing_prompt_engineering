# %%
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOllama # Using ChatOllama is often preferred

OLLAMA_MODEL = "llama3.1:8b" 

# --- 1. Define the Desired Output Structure using Pydantic ---
# This tells LangChain how to structure the final output.
class VideoScriptOutput(BaseModel):
    """Structure for the generated video script and headline."""
    headline: str = Field(description="Catchy, media-style headline (5-10 words) for the video.")
    video_script: str = Field(description="Engaging, promotional video script text (approx. 75-100 words, for 30-40 seconds runtime) including a hook, problem/opportunity explanation based on the news, solution proposal, and a clear call to action (CTA).")

# --- 2. Set up the Output Parser ---
# This parser will automatically convert the LLM's raw text output
# into the Pydantic model defined above.
parser = PydanticOutputParser(pydantic_object=VideoScriptOutput)

# --- 3. Define the Prompt Template ---
# This is where we adapt the original prompt for LangChain.
# We include instructions for the LLM and specify the desired output format
# using the parser's format instructions.

# We'll use ChatPromptTemplate for better structure, especially with chat models
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            """
            **Role:** You are an expert copywriter and marketer specializing in creating short, compelling video ad scripts for legal and consulting services. Your task is to turn potentially dry news into engaging content.

            **Context:** You will be given news text concerning bankruptcy procedures or recent legislative changes affecting businesses or individuals.

            **Task:** Based on the provided news text, generate materials for a 30-40 second promotional video. The goal is to capture the target audience's attention (entrepreneurs, affected citizens), explain the core issue or opportunity from the news, and motivate them to seek consultation or more information.

            **Input News Text:**
            ```
            {news_text}
            ```

            **Instructions & Required Output Structure:**

            1.  **Headline:** Create a catchy, intriguing media-style headline (5-10 words) that grabs attention immediately.
            2.  **Video Script (30-40 seconds / approx. 75-100 words):**
                * Start with a hook based on the news (e.g., a question or striking fact).
                * Briefly explain the essence of the news: What changed or what situation arose? What are the risks or opportunities for the viewer?
                * Suggest a solution: Hint at expert help, consultation, risk assessment, or subscribing for updates.
                * End with a clear and strong call to action (CTA) prompting the viewer to click a link, call, or submit a request.
                * The text should be dynamic, easy to understand when spoken.

            **Format your response strictly according to the following structure:**
            {format_instructions}
            """
        )
    ],
    input_variables=["news_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# --- 4. Instantiate the Ollama LLM ---
# Connects to the running Ollama instance.
# Using ChatOllama which is generally recommended.
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7) # Adjust temperature as needed

# --- 5. Create the LangChain Chain ---
# This links the prompt, LLM, and parser together.
# LCEL (LangChain Expression Language) syntax is used here (the pipe '|')
chain = prompt | llm | parser

# --- 6. Define Input News Data ---
# Example news text (same as before)
news_input = """
С 1 мая 2025 года вступают в силу поправки к закону о банкротстве физических лиц. Теперь процедура станет доступнее для граждан с долгом от 200 тысяч рублей (ранее порог был 500 тысяч). Однако ужесточаются требования к предоставлению документов и проверке сделок за последние 5 лет. Эксперты прогнозируют рост числа заявлений, но и увеличение отказов из-за ошибок при подаче.
"""

# --- 7. Run the Chain and Get Structured Output ---
try:
    print("--- Sending request to Ollama via LangChain ---")
    # Invoke the chain with the input news text
    result: VideoScriptOutput = chain.invoke({"news_text": news_input})

    print("\n--- Generated Output (Structured) ---")
    print(f"Headline: {result.headline}")
    print(f"\nVideo Script:\n{result.video_script}")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"Error: {e}")
    print("\nPlease ensure the Ollama server is running and the specified model ('{OLLAMA_MODEL}') is available.")
    print("You might need to run 'ollama pull {OLLAMA_MODEL}' in your terminal.")

# %%



