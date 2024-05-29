import pandas as pd
import streamlit as st
import warnings
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
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

# tab1, tab2,tab3 = st.tabs(["ChatApp","Dashboard","DatabaseOverview"])

warnings.filterwarnings('ignore')
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
df_combined = pd.merge(df, df2[['CU_NUMBER', 'CU_NAME', 'STATE_CODE', 'Peer_Group']], on="CU_NUMBER", how="left")

selected_cu_name = "CIATDEL"
selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]


# st.sidebar.image(image, use_column_width=True)

col1, col2= st.columns(2)

selected_cu = "CITADEL"

display_card("Risk and Analysis", cols[0])
def generate_delinquency_per_member_plot(selected_cu_name, attribute,attribute_title, height=350, width=300, ):
    st.write(" ")
    st.write(" ")
    selected_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]

    # Extract the state name and peer group for labeling
    selected_state_name = selected_data['STATE_CODE'].values[0]
    selected_peer_group = selected_data['Peer_Group'].values[0]

    # Delinquency per member for the selected CU
    selected_cu_delinquency_per_member = (selected_data[attribute] / selected_data['NO OF MEMBERS']).mean()

    # State-wide average delinquency per member calculation
    state_data = df_combined[df_combined['STATE_CODE'] == selected_state_name]
    state_delinquency_per_member = (state_data[attribute] / state_data['NO OF MEMBERS']).mean()

    # Peer group-wide average delinquency per member calculation
    peer_group_data = df_combined[df_combined['Peer_Group'] == selected_peer_group]
    peer_group_delinquency_per_member = (peer_group_data[attribute] / peer_group_data['NO OF MEMBERS']).mean()

    # Visualization
    labels = [f"{selected_cu_name}", f" State Average", "Peer Group Average"]
    values = [selected_cu_delinquency_per_member, state_delinquency_per_member, peer_group_delinquency_per_member]

    fig = px.bar(
        x=labels, 
        y=values, 
        labels={'x': '', 'y': 'Delinquency Per Member'},
        text=values,
        color=labels,
        color_discrete_map={
            labels[0]: '#19618A', 
            labels[1]: '#005950', 
            labels[2]: '#09B39C'
        },
        height=height, 
        width=width
    )

    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(title='Delinquency Per Member'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20),
        title=f"Delinquency Per Member: {attribute_title}"
    )
    
    fig.update_traces(
        texttemplate='%{y:.2f}', 
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>%{y:.2f}'
    )

    st.plotly_chart(fig, use_container_width=True)

def generate_delinquency_rate_plot(selected_cu_name, attribute,attribute_title, height=350, width=300, ):
    st.write(" ")
    st.write(" ")
    selected_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]

    # Extract the state name and peer group for labeling
    selected_state_name = selected_data['STATE_CODE'].values[0]
    selected_peer_group = selected_data['Peer_Group'].values[0]

    # Delinquency per member for the selected CU
    selected_cu_delinquency_per_member = (selected_data[attribute] / selected_data['Total number of loans and leases']).mean()

    # State-wide average delinquency per member calculation
    state_data = df_combined[df_combined['STATE_CODE'] == selected_state_name]
    state_delinquency_per_member = (state_data[attribute] / state_data['Total number of loans and leases']).mean()

    # Peer group-wide average delinquency per member calculation
    peer_group_data = df_combined[df_combined['Peer_Group'] == selected_peer_group]
    peer_group_delinquency_per_member = (peer_group_data[attribute] / peer_group_data['Total number of loans and leases']).mean()

    # Visualization
    labels = [f"{selected_cu_name}", f"Average in {selected_state_name}", "Peer Group Average"]
    values = [selected_cu_delinquency_per_member, state_delinquency_per_member, peer_group_delinquency_per_member]

    fig = px.bar(
        x=labels, 
        y=values, 
        labels={'x': '', 'y': 'Delinquency rate %'},
        text=values,
        color=labels,
        color_discrete_map={
            labels[0]: '#19618A', 
            labels[1]: '#005950', 
            labels[2]: '#09B39C'
        },
        height=height, 
        width=width
    )

    # Improve layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(title='Delinquency Rate %'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20),
        title=f"Delinquency Rate %: {attribute_title}"
    )
    
    fig.update_traces(
        texttemplate='%{y:.2f}', 
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>%{y:.2f}'
    )

    st.plotly_chart(fig, use_container_width=True)


with col1:
    generate_delinquency_per_member_plot("CITADEL", 'Deliquent Credit card loans (Amount) 0-180','0-180 Days')
    generate_delinquency_rate_plot("CITADEL", 'Deliquent Credit card loans (Amount) 0-180','0-180 Days')

with col2:
    generate_delinquency_per_member_plot("CITADEL", 'Deliquent Credit card loans (Amount) 180+','180+ Days')
    generate_delinquency_rate_plot("CITADEL", 'Deliquent Credit card loans (Amount) 180+','180+ Days')
    
