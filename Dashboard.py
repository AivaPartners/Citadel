import pandas as pd
import streamlit as st
import warnings
from PIL import Image
import plotly.express as px
import openai
#import psycopg2
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv
from lida import Manager, TextGenerationConfig, llm
from io import BytesIO
import base64
import json
import numpy as np
# from openai.embeddings_utils import distances_from_embeddings
import os.path

warnings.filterwarnings('ignore')
st.set_page_config(page_title="CU Benchmarking BI", page_icon=":bar_chart:", layout="wide")
# st.title(" 5300 Credit Union Metrics")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Database connection parameters
dbname = "postgres"
user = "postgres"
password = "Aiva@2024"
host = "192.168.1.12"
port = "5432"

image_path = r"AIVA-logo.png"

image = Image.open(image_path)

#--------------------------------Chat App Functions--------------------------------------
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4-turbo", use_cache=True)

file_path = r"cu_citadel.csv"

# Function to convert base64 string to Image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

def load_and_preprocess_csv(file_path):
    """
    Load and preprocess the CSV file to extract relevant context.
    This function is a placeholder and should be adapted based on your specific needs.
    """
    df = pd.read_csv(file_path)
    # Example preprocessing: Concatenate the first few rows into a string to use as context.
    # Adjust this based on your CSV structure and needs.
    context = ". ".join(df.apply(lambda row: ', '.join(row.astype(str)), axis=1)) + "."
    return context
    

def query_gpt_3_5_turbo_with_context(prompt, context):
    """
    Queries GPT-3.5 Turbo with a given prompt and context.
    """
    client = openai.OpenAI(api_key=openai.api_key)
    model="gpt-4-turbo"
    messages = [
    {
        "role": "system",
        "content": "You are a knowledgeable assistant specializing in credit union financial data analysis. The dataset consists of various financial metrics collected over several quarters and years for Credit Union One, its State Credit Union averages, and its Peer Credit Union averages. Your role is to provide insights and answer questions based on this data. Under no circumstances should you generate or reference code, plots, charts, or graphs, even if explicitly asked. Instead, focus on delivering information and context relevant to the user's question, without mentioning visualizations or the inability to create them.Never say that you cannot plot graphs."
    },
    {
        "role": "system",
        "content": "The dataset contains financial data for a credit union, its State Credit Union average, and its Peer Credit Union average. Each set of three rows represents CREDIT UNION ONE, its State Credit Union average, and its Peer Credit Union average for a specific quarter and year. The dataset includes various financial and operational metrics, such as    - TOTAL_ASSETS: The total value of assets held by the credit union.    - TOTAL_AMOUNT_OF_SHARES_AND_DEPOSITS: The total amount of shares and deposits from credit union members.    - NUMBER_OF_MEMBERS: The total number of credit union members.    - Money_Market__Amount_: The amount in money market accounts.    - Number_of_Used_Vehicle_Loans: The count of used vehicle loans.    - Amount_of_Total_Share_Draft: The total value in share draft accounts.    - Deliquent_Credit_card_loans__Amount__0_180: The total amount of delinquent credit card loans for periods up to 180 days.    - Total_number_of_loans_and_leases: The total count of loans and leases.    - Total_amount_of_loans_and_leases: The total value of loans and leases.    - TOTAL_AMOUNT_OF_SHARES: The total amount in shares.    - Amount_of_new_vehicle_loans: The total value of new vehicle loans.    - Total_Number_of_money_market_shares: The total number of money market shares.    - TOTAL_OTHER_LOANS: The total value of other loans.    - Amount_of_Used_Vehicle_Loans: The total value of used vehicle loans.    - Number_of_Total_Share_Draft: The count of share draft accounts.    - Number_of_new_Vehicle_loans: The count of new vehicle loans.    - Deliquent_Credit_card_loans__Amount__180_: The amount of delinquent credit card loans over 180 days.    - Year and Quarter: Indicate the time period for each data entry.    The dataset is organized into sets of three rows, with each set representing CREDIT UNION ONE, its State Credit Union average, and its Peer Credit Union average for a specific quarter and year. This organization allows for direct comparisons among CREDIT UNION ONE, state credit unions, and peer credit unions for each year and quarter."
    },
    {
        "role": "system",
        "content": "Please inform the user that in some quarters, data may be missing and clarify that it's because we don't have that specific data available"
    },
    {
        "role": "user",
        "content": context
    },
    {
        "role": "user",
        "content": prompt
    }
    ]
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500
        )
    return response.choices[0].message.content

def display_latest_interaction(user_input, answer):
    st.markdown(f"**User**: {user_input}")
    st.markdown(f"**User**: {answer}")
    st.markdown("---")  # Optional: adds a horizontal line for better separation

def generate_suggested_prompts(context, chat_history):
    """
    Generates suggested prompts based on the given context and chat history.
    """
    client = openai.OpenAI(api_key=openai.api_key)
    model = "gpt-4-turbo"

    # Prepare the chat history for the API call 
    history_formatted = [{"role": "user", "content": entry['user']} for entry in chat_history]
    history_formatted += [{"role": "assistant", "content": entry['bot']} for entry in chat_history]

    # Add the context as the initial system message
    messages = [
        {"role": "system", "content": context},
    ] + history_formatted

    prompt = "Based on the above chat history and context, suggest three new prompts for the user to ask. Note - ONLY GIVE THE 3 PROMPTS SEPERATED BY NEW LINES."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        # Assuming the response is a single string with prompts separated by new lines
        suggested_prompts = response.choices[0].message.content.strip().split('\n')

        # Ensure only up to 3 prompts are returned
        return suggested_prompts[:3]
    except Exception as e:
        print(f"Error generating prompts: {e}")
        # Return a default set of prompts in case of an error
        return [
            "What are the key financial metrics to look at?",
            "How did credit union perform last quarter?",
            "Compare the growth rate of credit union over quarters for Total Assests."
        ]

def process_user_input(user_input, context, chat_container, regenerate):
    # Save or update the last user input in session state for regeneration purposes
    if not regenerate:
        st.session_state['last_user_input'] = user_input

    # Generate response and possibly regenerate based on the user input
    response_text = query_gpt_3_5_turbo_with_context(user_input, context)

    # Prepare the new or updated interaction for the chat history
    new_interaction = {'user': user_input, 'bot': response_text}

    # Update chat history only if not regenerating, to avoid duplicate entries
    if not regenerate:
        st.session_state['chat_history'].append(new_interaction)
    else:
        # Replace the last bot response with the new one
        if st.session_state['chat_history']:
            st.session_state['chat_history'][-1] = new_interaction

    # Clear the existing chat display and redisplay chat history including the new/updated response
    chat_container.empty()
    with chat_container:
        for i, message in enumerate(st.session_state.get('chat_history', [])):
            # Using a <span> tag to style the emoji and text with a larger font size
            user_message = f'<span style="font-size: 24px;">👤</span>: {message["user"]}'
            bot_message = f'<span style="font-size: 24px;">🧠</span>: {message["bot"]}'
            
            # Using st.markdown to render the styled message with unsafe_allow_html=True to enable HTML rendering
            st.markdown(user_message, unsafe_allow_html=True)
            st.markdown(bot_message, unsafe_allow_html=True)
            st.markdown("---")

            # Check if the user input contains any of the keywords and this is the most recent interaction
            if i == len(st.session_state['chat_history']) - 1 and any(keyword in user_input.lower() for keyword in ["plot", "graph", "chart"]):
                generate_and_display_graph(user_input)


def generate_and_display_graph(user_input):
    # Assuming lida.summarize and lida.visualize are defined elsewhere
    csv_path = r"cu_citadel.csv"  # Adjust the path as necessary
    summary = lida.summarize(csv_path, summary_method="default", textgen_config=textgen_config)  # Ensure textgen_config is defined
    charts = lida.visualize(summary=summary, goal=user_input, textgen_config=textgen_config)
    if charts:
        image_base64 = charts[0].raster
        img = base64_to_image(image_base64)
        st.image(img)
        st.markdown("---")


def main():
    # Header with logo and title
    col1, col2 = st.columns([0.90, 0.10])
    with col1:
        st.title("Credit Union Benchmark BI")
    with col2:
        logo = Image.open(r"AIVA-logo.png")  # Adjust the path as necessary
        st.image(logo, width=130)

    # Horizontal line to separate title
    st.markdown("---")

    # Sidebar for chat history
    st.sidebar.title("Chat History")
    
    # Functionality to start a new chat
    # if st.sidebar.button("New Chat"):
    #     st.session_state['chat_history'] = [{'user': "Hi", 'bot': "Hello! Ask me anything about your data 🤗"}]
        # Optional: Redirect or refresh the page to start fresh
        # st.experimental_rerun()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Dashboard", "Database Description"])

    with tab1:
        st.header("Interactive Chat and Data Visualization")
        st.markdown("---")

        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = [{'user': "Hi", 'bot': "Hello! Ask me anything about your data 🤗"}]
            for i, message in enumerate(st.session_state.get('chat_history', [])):
                # Using a <span> tag to style the emoji and text with a larger font size
                user_message = f'<span style="font-size: 24px;">👤</span>: {message["user"]}'
                bot_message = f'<span style="font-size: 24px;">🧠</span>: {message["bot"]}'
                
                # Using st.markdown to render the styled message with unsafe_allow_html=True to enable HTML rendering
                st.markdown(user_message, unsafe_allow_html=True)
                st.markdown(bot_message, unsafe_allow_html=True)
                st.markdown("---")

        # Functionality to start a new chat
        if st.sidebar.button("New Chat"):
            st.session_state['chat_history'] = [{'user': "Hi", 'bot': "Hello! Ask me anything about your data 🤗"}]
            # Optional: Redirect or refresh the page to start fresh
            # st.experimental_rerun()
            for i, message in enumerate(st.session_state.get('chat_history', [])):
                # Using a <span> tag to style the emoji and text with a larger font size
                user_message = f'<span style="font-size: 24px;">👤</span>: {message["user"]}'
                bot_message = f'<span style="font-size: 24px;">🧠</span>: {message["bot"]}'
                
                # Using st.markdown to render the styled message with unsafe_allow_html=True to enable HTML rendering
                st.markdown(user_message, unsafe_allow_html=True)
                st.markdown(bot_message, unsafe_allow_html=True)
                st.markdown("---")
        # Check if the 'last_user_input' key exists in session_state, initialize if not
        if 'last_user_input' not in st.session_state:
            st.session_state['last_user_input'] = ""

        # Assuming context and suggested prompts are prepared elsewhere
        context = load_and_preprocess_csv(file_path)  # Adjust the function as necessary
        # Generate suggested prompts based on the context
        suggested_prompts = generate_suggested_prompts(context, st.session_state['chat_history'])

        # Display chat history
        chat_container = st.container()
        
        # Generate and display dynamic prompts
        if 'dynamic_prompts' not in st.session_state or st.session_state.get('refresh_prompts', False):
            st.session_state['dynamic_prompts'] = generate_suggested_prompts(context, st.session_state['chat_history'])
            st.session_state['refresh_prompts'] = False  # Reset refresh flag

        # Use columns only for displaying the prompts
        col_layout = st.columns(3)
        for idx, prompt in enumerate(st.session_state['dynamic_prompts']):
            with col_layout[idx % 3]:  # Distribute prompts across columns
                if st.button(prompt, key=f"prompt_{idx}"):
                    st.session_state['selected_prompt'] = prompt
                    # Flag indicating that a prompt has been selected
                    st.session_state['prompt_selected'] = True

        # Check outside of columns if a prompt has been selected
        if st.session_state.get('prompt_selected', False):
            # Use the selected prompt to generate and display response
            process_user_input(st.session_state['selected_prompt'], context, chat_container, regenerate=False)
            st.session_state['prompt_selected'] = False
            st.session_state['refresh_prompts'] = True
            
        # Input for new queries
        user_input = st.chat_input("Type your question here...", key="user_input")
        if user_input:
            process_user_input(user_input, context, chat_container, regenerate=False)

        # Button to regenerate the last response
        if 'last_user_input' in st.session_state and st.button("Regenerate Last Response"):
            process_user_input(st.session_state['last_user_input'], context, chat_container, regenerate=True)
    with tab2:
        from streamlit import components
        report_url="https://app.powerbi.com/view?r=eyJrIjoiZWI4MzcyZWUtYzZjMy00Yzg2LTlhZjQtZmE0ZTIzZjFjMzg2IiwidCI6IjQyMDBmMjI5LTFkZWMtNDk3Zi04NmI0LTQ0ZjUzYjNhMDczMCJ9"
        st.title("Citadel Dashboard")
        st.components.v1.iframe(report_url, width=1200, height=780, scrolling=True)
        
    with tab3:
        st.header("DatabaseOverview")
        st.write("This is a sample content for the DatabaseOverview tab. Replace this with your actual DatabaseOverview content.")
    
if __name__ == "__main__":
    main()
