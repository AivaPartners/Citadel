import pandas as pd
import streamlit as st
import warnings
from PIL import Image
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CU Benchmarking BI", page_icon=":bar_chart:", layout="wide")

# st.title(" 5300 Credit Union Metrics")
# st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Database connection parameters
# dbname = "postgres"
# user = "postgres"
# password = "Aiva@2024"
# host = "192.168.1.12"
# port = "5432"




# ------------------------------------------------------------------------------------------------
# Page Layout and Visuals

app_title = "Credit Union Benchmarking BI"

image_path = r"AIVA-logo.png"

# Open the image using PIL
image = Image.open(image_path)

# Display the image using Streamlit

# Using columns to layout the title and the logo
col1, col2 = st.columns([8,2])  # Adjust the ratio as needed for your layout

with col1:
    st.markdown(f"### {app_title}", unsafe_allow_html=True)

with col2:
    st.image(image, width=130) 

    
# @st.cache_data(hash_funcs={psycopg2.extensions.connection: id})
def load_data_from_db(file_path1,file_path2):
    df = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    return df, df2

cols = st.columns(4)

def display_card(title, column):
    column.markdown(f"""
    <div style="padding: 10px; border-radius: 10px; border: 1px solid #ccc; 
        box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
        text-align: center; justify-content:center;
        background-color:#EAEAEA;
        color: rgb(251, 250, 218);
        ">
        <h4 style="margin: 0; padding: 0; ">{title}</h4>
        
    </div>
    """, unsafe_allow_html=True)



file_paths1 = r'10Y+latQ.csv'
file_paths2 = r'FOICU.txt'
file_paths3 = r'ATM Locations.csv'
# df, df2 = load_data_from_db(file_path1,file_path2)
df = pd.read_csv(file_paths1)
df2 = pd.read_csv(file_paths2)
ATM_Locations = pd.read_csv(file_paths3, encoding='ISO-8859-1')
df_combined = pd.merge(df, df2[['CU_NUMBER', 'CU_NAME', 'STATE', 'Peer_Group']], on="CU_NUMBER", how="left")

selected_cu_name = "CIATDEL"
selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]



col1, col2, col3, col4= st.columns(4)
display_card("Growth Over Quarters", col1)

def get_previous_quarter(current_quarter, available_quarters):
    current_index = available_quarters.index(current_quarter)
    if current_index == 0:
        # If the current quarter is the earliest available quarter, return the current quarter itself
        return current_quarter
    else:
        # Otherwise, return the quarter before the current quarter
        return available_quarters[current_index - 1]

def calculate_growth_rate(current_value, previous_value):
    if previous_value == 0:
        return np.nan
    else:
        return ((current_value - previous_value) / previous_value) * 100

def generate_growth_rate_graph(selected_cu_name, attribute):
# Filter data for the selected CU
    selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]

    # Get unique quarters for the selected CU and sort them in descending order
    available_quarters = sorted(selected_cu_data['Quarter'].unique(), reverse=True)

    # Select the most recent 6 quarters (or less if there are fewer available)
    recent_quarters = available_quarters[:6]

    # Calculate growth rate for the selected CU using recent quarters
    growth_rate_cu = []
    for quarter in recent_quarters:
        current_value = selected_cu_data[selected_cu_data['Quarter'] == quarter][attribute].values[0]
        previous_quarter = get_previous_quarter(quarter, available_quarters)
        previous_value = selected_cu_data[selected_cu_data['Quarter'] == previous_quarter][attribute].values[0]
        growth_rate = calculate_growth_rate(current_value, previous_value)
        growth_rate_cu.append(growth_rate)

    # Calculate average growth rate for the state using recent quarters
    selected_state = selected_cu_data['STATE'].values[0]
    state_data = df_combined[df_combined['STATE'] == selected_state]
    growth_rate_state = []
    for quarter in recent_quarters:
        current_value_state = state_data[state_data['Quarter'] == quarter][attribute].mean()
        previous_quarter = get_previous_quarter(quarter, available_quarters)
        previous_value_state = state_data[state_data['Quarter'] == previous_quarter][attribute].mean()
        growth_rate_state.append(calculate_growth_rate(current_value_state, previous_value_state))

    # Calculate average growth rate for the peer group using recent quarters
    selected_peer_group = selected_cu_data['Peer_Group'].values[0]
    peer_data = df_combined[df_combined['Peer_Group'] == selected_peer_group]
    growth_rate_peer = []
    for quarter in recent_quarters:
        current_value_peer = peer_data[peer_data['Quarter'] == quarter][attribute].mean()
        previous_quarter = get_previous_quarter(quarter, available_quarters)
        previous_value_peer = peer_data[peer_data['Quarter'] == previous_quarter][attribute].mean()
        growth_rate_peer.append(calculate_growth_rate(current_value_peer, previous_value_peer))

    # Convert quarters to the format "Q-YY" for x-axis labels
    quarter_labels = [f"Q{int(q.split('-')[1]) % 4 + 1}-{q.split('-')[0][-2:]}" for q in reversed(recent_quarters)]
    # Reverse growth rate lists to match the correct order
    growth_rate_cu.reverse()
    growth_rate_state.reverse()
    growth_rate_peer.reverse()

    # Plot the growth rate graphs
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_cu, mode='lines+markers', name=selected_cu_name))
    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_state, mode='lines+markers', name=f"{selected_state} State Avg"))
    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_peer, mode='lines+markers', name=f"{selected_peer_group} Peer Avg"))
    fig.data[0].name = "CU"

    fig.update_layout(
        title=f"{attribute} Growth Rate",
        xaxis_title="Quarter",
        yaxis_title="Growth Rate (%)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)




def generate_multiple_growth_rate_graphs(selected_cu_name):
# Get the first 6 attributes from the dataframe
    attributes = df_combined.columns[1:7]  # Exclude the first column which is 'quarter'

    # Organize attributes into two rows with three graphs each
    num_columns = 3  # Number of columns in each row
    num_rows = 2  # Number of rows

    for i in range(num_rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            index = i * num_columns + j
            if index < len(attributes):
                with cols[j]:
                    # st.header(f"{attributes[index]} Growth Rate")
                    generate_growth_rate_graph(selected_cu_name, attributes[index])




# Call the function to generate multiple growth rate graphs


# Example usage
st.write(" ")
selected_cu_name = "CITADEL"  # Assuming this is the CU of focus
generate_multiple_growth_rate_graphs(selected_cu_name)

    







