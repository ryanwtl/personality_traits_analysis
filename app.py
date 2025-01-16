import streamlit as st
import pandas as pd
import io
from pydub import AudioSegment
from pydub.playback import play
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from groq import Groq
import os
from langchain_core.prompts import ChatPromptTemplate

# Title of the app
st.title("Personality Traits Analysis Streamlit App")

# Predefined Variables

client = Groq()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
personality_traits = {}
count = 0

traits = ['cOPN','cCON','cEXT','cAGR','cNEU']
full_trais = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
for trait in traits:
    personality_traits[trait]=full_trais[count]
    count += 1

def load_roberta(model_path,device):
    # Load the saved model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, token="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", num_labels=1)
    model.to(device)
    return model,tokenizer

def load_analysis_engine():
    # Load the saved model and tokenizer
    model_path = "theweekday/personality_traits_"
    models = {}
    tokenizers = {}

    for t,trait in personality_traits.items():
        path = f"{model_path}{trait}"
        models[t], tokenizers[t] = load_roberta(path, "cuda")
    return models, tokenizers

def text_personality_traits_analysis(sentence,models,tokenizers,max_length=512):
    st.write(f"""You entered : \n
             '{content}' """)
    # Function to analyze the personality traits from text
    analysis = {}
    components = ['value','score']
    for trait, model in models.items():
        encodings = tokenizers[trait]([sentence], truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        model.eval()
        with torch.no_grad():
            output = {}
            score = torch.sigmoid(model(**encodings).logits).item()
            binary_value = 'y' if score > 0.61 else 'n'
            # output = f"{binary_value}, score : {score:.4f}"
            output[components[0]] = binary_value
            output[components[1]] = f"{score:.4f}"
            # if(trait!='cNEU'):
            #     output += ","
            analysis[personality_traits[trait]]=output

    return analysis

def csv_personality_traits_analysis(content):
    # Function to analyze the personality traits from CSV
    st.write("Sorry, this feature is not available yet.")

def mp3_to_text(content):
    # Function to convert MP3 to text
    st.write("Sorry, this feature is not available yet.")

def analysis_result_output1(analysis):
    # Function to output the analysis result
    st.write("### Personality Traits Analysis Result:")

    for trait, details in analysis.items():
        value = details['value']
        score = float(details['score'])  # Convert score to float for comparison or display formatting
        
        # Define color based on the score (example: higher score = more positive)
        if(value == 'y'):
            if score > 0.6:
                color = "green"
            elif score > 0.5:
                color = "yellow"
        else:
            color = "red"

        # Display each trait with a formatted structure
        st.markdown(f"**{trait.capitalize()}**:")
        st.write(f"  - **Value**: {'Yes' if value == 'y' else 'No'}")
        st.write(f"  - **Score**: {score:.4f}", unsafe_allow_html=True)
        
        # Displaying score with color highlighting (green/yellow/red)
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>Score: {score:.4f}</span>", unsafe_allow_html=True)
        st.write("---")  # Separator for each trait

def analysis_result_output2(analysis):
    # Create a list of dictionaries to store the results for each trait
    results = []
    
    for trait, details in analysis.items():
        value = 'Yes' if details['value'] == 'y' else 'No'
        score = float(details['score'])
        
        # Append the data to the results list
        results.append({
            "Trait": trait.capitalize(),
            "Value (Yes/No)": value,
            "Score": f"{score:.4f}"
        })
    
    # Convert the results to a DataFrame
    df = pd.DataFrame(results)
    
    # Display the table in Streamlit
    st.write("### Personality Traits Analysis Result:")
    st.dataframe(df, use_container_width=True)

def generate_query_messages(content):
    """Create the messages payload that will be sent to the Groq LLM API."""
    return [
        {
            "role": "system",
            "content": """You are an assistant for providing insights based on the personality traits score.
                        Use the following 5 traits of personality scores and provide insights based on the scores.
                        """
        },
        {
            "role": "user",
            "content": content
        }
    ]

# Prompt template
PROMPT_TEMPLATE = """
If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Provide insights based on the above context.
"""

def insights_of_results(analysis):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=analysis)

    messages = generate_query_messages(prompt)
    print(f"this is the message : \n{messages}")

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Ensure this model name is correct
        messages=messages,
        temperature=1,
        max_tokens=500,
        top_p=1,
        stream=False,
    )

    result = response.choices[0].message.content

    st.write(result)

# ======================================= Analysis Starts HERE ======================================= #
# Sidebar for navigation
option = st.sidebar.selectbox("Choose an option:", ["Prompt a Sentence", "Upload a CSV", "Upload an MP3"])

models, tokenizers = load_analysis_engine()

if option == "Prompt a Sentence":
    # Sentence input and display
    content = st.text_input("Enter a sentence:")
    if content:
        analysis = text_personality_traits_analysis(content,models,tokenizers)
        analysis_result_output2(analysis)
        insights_of_results(analysis)
    

elif option == "Upload a CSV":
    # File uploader for CSV
    content = st.file_uploader("Upload a CSV file", type=["csv"])
    if content:
        df = pd.read_csv(content, encoding='latin-1')
        st.write("Content of the uploaded CSV:")
        st.dataframe(df)
        csv_personality_traits_analysis(content)

elif option == "Upload an MP3":
    # File uploader for MP3
    content = st.file_uploader("Upload an MP3 file", type=["mp3"])
    if content:
        # Play the uploaded MP3 file
        st.audio(content, format="audio/mp3")
        mp3_to_text(content)

# # Check and display the data type of the content variable
# if content is not None:
#     st.write("Data type of content:", type(content))
