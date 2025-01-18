import streamlit as st
import pandas as pd
import io
import time
import os
from pydub import AudioSegment
from pydub.playback import play
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq
from dotenv import load_dotenv
import matplotlib.image as mpimg

system_start_time = time.time()
# Set the page layout to wide mode
st.set_page_config(layout="wide")
# Title of the app
st.title("Personality Traits Analysis Streamlit App")

# Predefined Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
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
    tokenizer = RobertaTokenizer.from_pretrained(model_path, token=huggingface_api_key)
    # model = RobertaForSequenceClassification.from_pretrained(model_path, token="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", num_labels=1)
    model = RobertaForSequenceClassification.from_pretrained(model_path, token=huggingface_api_key, num_labels=1)
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

def model_evaluation():
    # Subheader for the model evaluation section
    st.subheader('RoBERTa Model Performance on BIG 5 Personality Traits')

    # Create two columns for the table and image to be displayed side by side
    col1, col2 = st.columns([1, 1])  # You can adjust the proportions as needed

    with col1:

        img = mpimg.imread('.\results\[BIG5]_RoBERTa_ver4.png')
        st.caption('AUC-ROC Curve for RoBERTa Model')
        st.image(img,  use_container_width=True)

    with col2:
        df = pd.read_csv(".\results\performance_roberta.csv")

        # Show the table
        st.caption('Performance Table of RoBERTa model on BIG 5')
        st.write('')
        st.table(df)
        
    # Add the performance analysis text under the table
    st.write("""
    Refer to the **AUC-ROC Curve Graph** and the **Performance Table**, there are a few key points to highlight:

1. **Overall AUC-ROC Scores**:
   - **Openness (cOPN)** stands out with the highest AUC of 0.78, indicating the model has a strong ability to differentiate between classes for this trait.
   - The remaining traits (**Conscientiousness (cCON)**, **Extraversion (cEXT)**, **Agreeableness (cAGR)**, and **Neuroticism (cNEU)**) have more modest AUC values ranging from 0.59 to 0.64, indicating weaker performance in separating positive and negative cases.

2. **Performance Highlights**:
   - The high AUC for **Openness** aligns with its perfect recall score (1.0) from the earlier table, showing the model effectively captures true positives. However, its low precision (0.5142) suggests the model struggles with false positives, which may limit practical usability despite its high AUC.
   - Traits like **Neuroticism** (AUC 0.59) and **Agreeableness** (AUC 0.62) are the weakest performers in terms of AUC, reflecting their lower overall balance between recall, precision, and F1 scores.

3. **Model Behavior**:
   - The steep initial rise in the ROC curve for **Openness** indicates better performance at lower false positive rates compared to other traits.
   - Traits such as **Agreeableness** and **Neuroticism** show ROC curves closer to the diagonal (random guessing), suggesting the model struggles to predict these traits accurately.

### In a nutshell:
The AUC-ROC results confirm the strengths and weaknesses seen in the other performance metrics. While **Openness** shows relatively strong performance, other traits—especially **Agreeableness** and **Neuroticism**—require further optimization to improve both the AUC and other metrics like precision and accuracy. The model may benefit from more training data, better feature engineering, or hyperparameter tuning to achieve better class separation for the lower-performing traits.
             """)

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

Provide insights based on the above context. Start with the '1. Openness : '.
"""

def insights_of_results(analysis):
    model = "llama-3.1-70b-versatile"
    temperature = 1.0
    max_tokens = 500
    top_p = 1

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=analysis)

    messages = generate_query_messages(prompt)
    print(f"this is the message : \n{messages} \n")

    response = client.chat.completions.create(
        model=model,  # Ensure this model name is correct
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
    )

    result = response.choices[0].message.content
    st.write(f"""Respond from Chat Assistant : {model}""")
    st.write(f"""Based on the provided personality traits scores, here are some insights : \n{result}""")

# ======================================= Analysis Starts HERE ======================================= #

models, tokenizers = load_analysis_engine()
system_elapsed_time = time.time() - system_start_time
print(f"\nSystem Elapsed time: {system_elapsed_time:.4f} seconds \n")

model_evaluation()

# Sidebar for navigation
option = st.sidebar.selectbox("Choose an option:", ["Prompt a Sentence For Demo", "Recommendation Analysis"])

if option == "Prompt a Sentence For Demo":
    # Sentence input and display
    content = st.text_input("Enter a sentence for personality traits analysis:")
    start_time = time.time()
    if content:
        analysis = text_personality_traits_analysis(content,models,tokenizers)
        analysis_result_output2(analysis)
        insights_of_results(analysis)

        elapsed_time = time.time() - start_time
        print(f"Response Elapsed time: {elapsed_time:.4f} seconds \n")
    

elif option == "Recommendation Analysis":
    # File uploader for CSV
    content = st.file_uploader("Upload a CSV file", type=["csv"])
    start_time = time.time()
    if content:
        df = pd.read_csv(content, encoding='latin-1')
        st.write("Content of the uploaded CSV:")
        st.dataframe(df)
        csv_personality_traits_analysis(content)

        elapsed_time = time.time() - start_time
        print(f"Response Elapsed time: {elapsed_time:.4f} seconds \n")
