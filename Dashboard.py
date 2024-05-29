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
            temperature=0
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
            "How did credit union 61650 perform last quarter?",
            "Compare the growth rate of credit unions 61650 and 61466."
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

def calculate_growth_rate(df, attribute, group, group_value, quarter_1, quarter_2):
    # Filter the data for the selected group in the given quarters
        filtered_df = df[(df[group] == group_value) & (df['Quarter'].isin([quarter_1, quarter_2]))]
        
        # Get the attribute values for the two quarters
        value_1 = filtered_df[filtered_df['Quarter'] == quarter_1][attribute].mean()
        value_2 = filtered_df[filtered_df['Quarter'] == quarter_2][attribute].mean()

        # Calculate the growth rate and round it to 2 decimal places
        growth_rate = round(((value_2 - value_1) / value_1) * 100, 2) if value_1 != 0 else 0
        return growth_rate
def calculate_cu_growth_rate(df, attribute, cu_name, quarter_1, quarter_2):
    # Filter the data for the selected CU in the given quarters
    filtered_df = df[(df['CU_NAME'] == cu_name) & (df['Quarter'].isin([quarter_1, quarter_2]))]

    # Ensure the data is correctly filtered and we have the values for both quarters
    if filtered_df['Quarter'].nunique() == 2:
        # Get the attribute values for the two quarters
        value_1 = filtered_df[filtered_df['Quarter'] == quarter_1][attribute].iloc[0]
        value_2 = filtered_df[filtered_df['Quarter'] == quarter_2][attribute].iloc[0]

        # Calculate the growth rate and round it to 2 decimal places
        growth_rate = round(((value_2 - value_1) / value_1) * 100, 2) if value_1 != 0 else 0
    else:
        # If we don't have both values, we cannot calculate the growth rate
        growth_rate = None
    
    return growth_rate
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
    
            def load_data_from_db(file_path1,file_path2):
                df = pd.read_csv(file_path1)
                df2 = pd.read_csv(file_path2)
                return df, df2

            cols = st.columns(4)
            def display_card(title, value, column):
                column.markdown(f"""
                <div style="padding: 10px; border-radius: 10px; border: 1px solid #ccc; 
                    box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
                    text-align: center; justify-content:center;
                    background-color:#EAEAEA;
                    color: rgb(251, 250, 218);
                    ">
                    <h4 style="margin: 0; padding: 0; ">{title}</h4>
                    <h3 style="margin: 0; padding: 10px 0;">{value}</h3>
                </div>
                """, unsafe_allow_html=True)
            def human_readable_number(num):
                num = float(num)
                magnitude = 0
                while abs(num) >= 1000:
                    magnitude += 1
                    num /= 1000.0
                # Add more suffixes if you need them
                return f'{num:.2f}{" KMBT"[magnitude]}'
            def inject_custom_button_css():
                custom_css = """
                <style>
                /* Custom CSS for Streamlit buttons */
                .stButton>button {
                align-items: center;
                appearance: none;
                background-color: 436850;
                border-radius: 4px;
                border-width: 0;
                box-shadow: rgba(45, 35, 66, 0.4) 0 2px 4px,
                            rgba(45, 35, 66, 0.3) 0 7px 13px -3px,
                            #D6D6E7 0 -3px 0 inset;
                box-sizing: border-box;
                color: 12372A;
                cursor: pointer;
                display: inline-flex;
                font-family: "JetBrains Mono", monospace;
                height: 48px;
                justify-content: center;
                line-height: 1;
                overflow: hidden;
                padding-left: 16px;
                padding-right: 16px;
                position: relative;
                text-align: left;
                text-decoration: none;
                transition: box-shadow .15s, transform .15s;
                user-select: none;
                -webkit-user-select: none;
                touch-action: manipulation;
                white-space: nowrap;
                will-change: box-shadow, transform;
                font-size: 18px;
                }
                .stButton>button:focus {
                box-shadow: #D6D6E7 0 0 0 1.5px inset,
                            rgba(45, 35, 66, 0.4) 0 2px 4px,
                            rgba(45, 35, 66, 0.3) 0 7px 13px -3px,
                            #D6D6E7 0 -3px 0 inset;
                }
                .stButton>button:hover {
                box-shadow: rgba(45, 35, 66, 0.4) 0 4px 8px,
                            rgba(45, 35, 66, 0.3) 0 7px 13px -3px,
                            #D6D6E7 0 -3px 0 inset;
                transform: translateY(-2px);
                }
                .stButton>button:active {
                box-shadow: #D6D6E7 0 3px 7px inset;
                transform: translateY(2px);
                }
                </style>
                """
                st.markdown(custom_css, unsafe_allow_html=True)
            file_paths1 = r'10YearDatabase.csv'
            file_paths2 = r'FOICU.txt'
            file_paths3 = r'ATM Locations.csv'
            # df, df2 = load_data_from_db(file_path1,file_path2)
            df = pd.read_csv(file_paths1)
            df2 = pd.read_csv(file_paths2)
            ATM_Locations = pd.read_csv(file_paths3, encoding='ISO-8859-1')
            df_combined = pd.merge(df, df2, on="CU_NUMBER", how="left")

            selected_cu_name = "CITADEL"
            selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
            
            st.header(f"Dashboard: CITADEL")
            # st.header(f"Dashboard: {{CU1}}")
        
            # Sidebar with CU_NAME selection
            col1, col2, col3, col4= st.columns(4)# Place the selectbox in the first column
            
            total_assets = selected_cu_data['TOTAL ASSETS'].values[0]
            total_sharedraft = selected_cu_data['Amount of Total Share Draft'].values[0]
            no_of_members = selected_cu_data['NO OF MEMBERS'].values[0]
            total_loans = selected_cu_data['Total amount of loans and leases'].values[0]
            total_deposits = selected_cu_data['TOTAL AMOUNT OF SHARES AND DEPOSITS'].values[0]
            
            total_assets_formatted = f"${human_readable_number(total_assets)}"
            total_sharedraft_formatted = human_readable_number(total_sharedraft)  
            no_of_members_formatted = human_readable_number(no_of_members)
            total_loans_formatted = f"${human_readable_number(total_loans)}"
            total_deposits_formatted = f"${human_readable_number(total_deposits)}"

            # Display each metric in its respective card container
            metrics = [("Total Assets", total_assets_formatted), 
                    ("# of Members", no_of_members_formatted), 
                    ("Loans", total_loans_formatted), 
                    ("Deposits", total_deposits_formatted)]
            


            for col, (title, value) in zip([col1, col2, col3, col4], metrics):
                display_card(title, value, col)

            filtered_df_for_selected_cu = df_combined[df_combined['CU_NAME'] == selected_cu_name]
            available_quarters = sorted(filtered_df_for_selected_cu['Quarter'].unique())

            if len(available_quarters) >= 2:
                # Automatically select the quarter before the most recent quarter as quarter_1
                # and the most recent quarter as quarter_2
                quarter_1, quarter_2 = available_quarters[-2], available_quarters[-1]
            else:
                st.error("Not enough data available for the selected CU to compare quarters.")
                # Optionally, handle cases with less than 2 quarters available

            if 'selected_button' not in st.session_state:
                st.session_state.selected_button = ""

        # Create a row for buttons with 3 columns
            cols = st.columns(3)

        # Place each button in its own column with reduced space between them
            # inject_custom_button_css()
            with cols[0]:
                if st.button("Risk", key="btn2"):
                    st.session_state.selected_button = "Risk"

            with cols[1]:
                if st.button("Member Metrics", key="btn3"):
                    st.session_state.selected_button = "Member Metrics"

            with cols[2]:
                if st.button("Growth"):
                    st.session_state.selected_button = "Growth"

            if st.session_state.selected_button == "Member Metrics":
                attributes = ['TOTAL ASSETS', 'Total amount of loans and leases', 'NO OF MEMBERS']
                graphs_cols = st.columns(3)
                selected_quarter = available_quarters[0]
                selected_cu_name = 'CITADEL'
                selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
                selected_state = selected_cu_data['STATE'].iloc[0]
                selected_peer_group = selected_cu_data['Peer_Group'].iloc[0]        
                
                # Calculate Number of Members for CU, State, and Peer Group
                cu_members = selected_cu_data['NO OF MEMBERS'].iloc[0]
                state_avg_members = df_combined[df_combined['STATE'] == selected_state]['NO OF MEMBERS'].mean()
                peer_group_avg_members = df_combined[df_combined['Peer_Group'] == selected_peer_group]['NO OF MEMBERS'].mean()

                # Calculate Loans per Member for CU, State, and Peer Group
                cu_loans_per_member = selected_cu_data['Total amount of loans and leases'].iloc[0] / cu_members
                state_total_loans = df_combined[df_combined['STATE'] == selected_state]['Total amount of loans and leases'].sum()
                state_total_members = df_combined[df_combined['STATE'] == selected_state]['NO OF MEMBERS'].sum()
                state_avg_loans_per_member = state_total_loans / state_total_members
                peer_group_total_loans = df_combined[df_combined['Peer_Group'] == selected_peer_group]['Total amount of loans and leases'].sum()
                peer_group_total_members = df_combined[df_combined['Peer_Group'] == selected_peer_group]['NO OF MEMBERS'].sum()
                peer_group_avg_loans_per_member = peer_group_total_loans / peer_group_total_members

                # Calculate TOTAL ASSETS per Member for CU, State, and Peer Group
                cu_assets_per_member = selected_cu_data['TOTAL ASSETS'].iloc[0] / cu_members
                state_total_assets = df_combined[df_combined['STATE'] == selected_state]['TOTAL ASSETS'].sum()
                state_avg_assets_per_member = state_total_assets / state_total_members
                peer_group_total_assets = df_combined[df_combined['Peer_Group'] == selected_peer_group]['TOTAL ASSETS'].sum()
                peer_group_avg_assets_per_member = peer_group_total_assets / peer_group_total_members

                # Create dataframes for plotting
                data_members = pd.DataFrame({'Category': ['Citadel', 'State', 'Peer Group'], 'Value': [cu_members, state_avg_members, peer_group_avg_members]})
                data_loans_per_member = pd.DataFrame({'Category': ['Citadel', 'State', 'Peer Group'], 'Value': [cu_loans_per_member, state_avg_loans_per_member, peer_group_avg_loans_per_member]})
                data_assets_per_member = pd.DataFrame({'Category': ['Citadel', 'State', 'Peer Group'], 'Value': [cu_assets_per_member, state_avg_assets_per_member, peer_group_avg_assets_per_member]})

                # Plotting
                fig_members = px.bar(data_members, x='Category', y='Value', title='Number of Members Analysis')
                fig_loans_per_member = px.bar(data_loans_per_member, x='Category', y='Value', title='Loans per Member Analysis')
                fig_assets_per_member = px.bar(data_assets_per_member, x='Category', y='Value', title='Assets per Member Analysis')

                # Display the graphs in one single row with three columns

                fig_members.update_layout(
                            autosize=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title=f"{attributes[2]}"),
                            yaxis=dict(title='Members'),
                            # title=dict(text=f" ", x=0.5, xanchor='center'),
                            font=dict(size=16),
                            height=400,
                            hovermode=False            
                )
                fig_members.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                fig_loans_per_member.update_layout(
                            autosize=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title=f"{attributes[1]}"),
                            yaxis=dict(title='Ratio'),
                            # title=dict(text=f" from {quarter_1} to {quarter_2}", x=0.5, xanchor='center'),
                            font=dict(size=16),
                            height=400, 
                            hovermode=False            
                            )
                fig_loans_per_member.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                
                fig_assets_per_member.update_layout(
                            autosize=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title=f"{attributes[0]}"),
                            yaxis=dict(title='Ratio'),
                            # title=dict(text=f" from {quarter_1} to {quarter_2}", x=0.5, xanchor='center'),
                            font=dict(size=16),
                            height=400, 
                            hovermode=False    
                            )
                fig_assets_per_member.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                cols = st.columns(3)
                with cols[0]:
                    st.plotly_chart(fig_members, use_container_width=True)
                with cols[1]:
                    st.plotly_chart(fig_loans_per_member, use_container_width=True)
                with cols[2]:
                    st.plotly_chart(fig_assets_per_member, use_container_width=True)
            # Check which button was clicked and display graphs accordingly
            if st.session_state.selected_button == "Risk":
                attributes = ['Delinquent Credit card loans (Amount) 0-180', 'Delinquent Credit card loans (Amount) 180+']
                graph_cols = st.columns(3)
                selected_quarter = available_quarters[0]
                selected_cu_name = 'CITADEL'
                selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
                selected_state = selected_cu_data['STATE'].iloc[0]
                selected_peer_group = selected_cu_data['Peer_Group'].iloc[0]
                cu_members = selected_cu_data['Total amount of loans and leases'].iloc[0]
                state_total_members = df_combined[df_combined['STATE'] == selected_state]['Total amount of loans and leases'].sum()
                peer_group_total_members = df_combined[df_combined['Peer_Group'] == selected_peer_group]['Total amount of loans and leases'].sum()

                # Step 3: Calculate average Delinquent Credit card loans (Amount) 0-180 for all CU's in the state and peer group
                state_avg_0_180 = df_combined[df_combined['STATE'] == selected_state]['Deliquent Credit card loans (Amount) 0-180'].mean()
                state_avg_0_180 = state_avg_0_180
                peer_group_avg_0_180 = df_combined[df_combined['Peer_Group'] == selected_peer_group]['Deliquent Credit card loans (Amount) 0-180'].mean()
                peer_group_avg_0_180 = peer_group_avg_0_180
                cu_avg_0_180 = selected_cu_data['Deliquent Credit card loans (Amount) 0-180'].iloc[0]
                cu_avg_0_180 = cu_avg_0_180

                # Step 4: Calculate average Delinquent Credit card loans (Amount) 180+ for all CU's in the state and peer group
                state_avg_180_plus = df_combined[df_combined['STATE'] == selected_state]['Deliquent Credit card loans (Amount) 180+'].mean()
                state_avg_180_plus = state_avg_180_plus
                peer_group_avg_180_plus = df_combined[df_combined['Peer_Group'] == selected_peer_group]['Deliquent Credit card loans (Amount) 180+'].mean()
                peer_group_avg_180_plus = peer_group_avg_180_plus
                cu_avg_180_plus = selected_cu_data['Deliquent Credit card loans (Amount) 180+'].iloc[0]
                cu_avg_180_plus = cu_avg_180_plus
                # Step 5: Calculate average of ( Delinquent Credit card loans (Amount) 0-180 + Delinquent Credit card loans (Amount) 180+ )
                state_avg_total = (state_avg_0_180 + state_avg_180_plus) 
                state_avg_total = state_avg_total
                peer_group_avg_total = (peer_group_avg_0_180 + peer_group_avg_180_plus) 
                peer_group_avg_total = peer_group_avg_total
                cu_avg_total = (cu_avg_0_180 + cu_avg_180_plus)
                cu_avg_total = cu_avg_total
                data_0_180 = {
                    'Category': ['Citadel', 'State', 'Peer Group'],
                    'Value': [cu_avg_0_180, state_avg_0_180, peer_group_avg_0_180]
                }

                # Create data for the second bar graph (Delinquent Credit card loans (Amount) 180+)
                data_180_plus = {
                    'Category': ['Citadel', 'State', 'Peer Group'],
                    'Value': [cu_avg_180_plus, state_avg_180_plus, peer_group_avg_180_plus]
                }

                # Create data for the third bar graph (Average of Delinquent Credit card loans (Amount) 0-180 and Delinquent Credit card loans (Amount) 180+)
                data_total = {
                    'Category': ['Citadel', 'State', 'Peer Group'],
                    'Value': [cu_avg_total, state_avg_total, peer_group_avg_total]
                } 

                fig_layout = dict(
                    autosize=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title=''),
                    yaxis=dict(title='Delinquent Credit card loans Amount'),
                    font=dict(size=16),
                    height=400, 
                    hovermode=False    
                )

                fig_0_180 = px.bar(data_0_180, x='Category', y='Value', title=' 0-180 Delinquency ')
                fig_180_plus = px.bar(data_180_plus, x='Category', y='Value', title='180+ Delinquency ')
                fig_total = px.bar(data_total, x='Category', y='Value', title='Delinquency throughout the year')

                # Display the graphs in one single row with three columns
                fig_0_180.update_layout(**fig_layout, title=dict(x=0.5, xanchor='center'))
                fig_180_plus.update_layout(**fig_layout, title=dict(x=0.5, xanchor='center'))
                fig_total.update_layout(**fig_layout, title=dict(x=0.5, xanchor='center'))

                fig_0_180.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                fig_180_plus.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                fig_total.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                cols = st.columns(3)
                with cols[0]:
                    st.plotly_chart(fig_0_180, use_container_width=True)

                with cols[1]:
                    st.plotly_chart(fig_180_plus, use_container_width=True)

                with cols[2]:
                    st.plotly_chart(fig_total, use_container_width=True)
            if st.session_state.selected_button == "Growth" and len(available_quarters) >= 2:
                attributes = ['TOTAL ASSETS', 'Total amount of loans and leases', 'NO OF MEMBERS']
                
                # Create columns for each graph. The number of columns equals the number of attributes.
                graph_cols = st.columns(len(attributes))
                
                for index, attribute in enumerate(attributes):
                    cu_growth_rate = calculate_cu_growth_rate(df_combined, attribute, selected_cu_name, quarter_1, quarter_2)
                    selected_state_code = df_combined[df_combined['CU_NAME'] == selected_cu_name]['STATE'].iloc[0]
                    selected_peer_group = df_combined[df_combined['CU_NAME'] == selected_cu_name]['Peer_Group'].iloc[0]
                    state_growth_rate = calculate_growth_rate(df_combined, attribute, 'STATE', selected_state_code, quarter_1, quarter_2)
                    peer_group_growth_rate = calculate_growth_rate(df_combined, attribute, 'Peer_Group', selected_peer_group, quarter_1, quarter_2)

                    # Prepare data for plotting
                    growth_data = {
                        'Category': ['Citadel', 'State', 'Peer Group'],
                        'Growth Rate': [cu_growth_rate, state_growth_rate, peer_group_growth_rate]
                    }
                    growth_df = pd.DataFrame(growth_data)

                    # Plotting the bar chart within the specific column
                    with graph_cols[index]:
                        fig = px.bar(growth_df, x='Category', y='Growth Rate', title=f"{attribute} Growth Rate from {quarter_1} to {quarter_2}",
                                    labels={'Growth Rate': 'Growth Rate (%)'})
                        fig.update_layout(
                            autosize=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title=f"{attribute}"),
                            yaxis=dict(title='Growth Rate (%)'),
                            title=dict(text=f" from {quarter_1} to {quarter_2}", x=0.5, xanchor='center'),
                            font=dict(size=16),
                            height=400,
                            hovermode=False
                        )
                        fig.update_traces(marker_color=['#19618A', '#005950', '#09B39C'])
                        st.plotly_chart(fig, use_container_width=True)

            # Display which button is selected
            if st.session_state.selected_button:
                st.write(f"{st.session_state.selected_button} button clicked")
        # Content for DatabaseOverview tab
        with tab3:
            st.header("DatabaseOverview")
            st.write("This is a sample content for the DatabaseOverview tab. Replace this with your actual DatabaseOverview content.")
    
if __name__ == "__main__":
    main()
