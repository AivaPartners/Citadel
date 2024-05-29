import pandas as pd
import streamlit as st
import warnings
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
    st.write(" ")

file_paths1 = r'10Year+latQ.csv'
file_paths2 = r'FOICU.txt'
file_paths3 = r'ATM Locations.csv'
# df, df2 = load_data_from_db(file_path1,file_path2)
df = pd.read_csv(file_paths1)
df2 = pd.read_csv(file_paths2)
ATM_Locations = pd.read_csv(file_paths3, encoding='ISO-8859-1')

df_combined = pd.merge(df, df2[['CU_NUMBER', 'CU_NAME', 'STATE', 'Peer_Group']], on="CU_NUMBER", how="left")

selected_cu_name = "CITADEL"
selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]

no_of_ATMS = ATM_Locations[ATM_Locations['CU_NAME']==selected_cu_name].count
cols = st.columns(3)
display_card("Member Metrics", cols[0])

col1, col2= st.columns(2)



def generate_atm_locations_comparison_plot(selected_cu_name, ATM_Locations, df_combined):
    """
    Generates a plot comparing the number of ATMs for the selected credit union
    against the state average and peer group average.
    
    Parameters:
    - selected_cu_name: Name of the selected credit union.
    - ATM_Locations: DataFrame containing ATM locations, expected to have
                    'CU_NAME' and 'PhysicalAddressStateCode' columns.
    - df_combined: DataFrame containing credit union details, expected to have
                'CU_NAME', 'STATE_CODE', and 'Peer_Group' columns.
    """
    # Extract the state name and peer group for the selected CU
    selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
    selected_state_name = selected_cu_data['STATE'].iloc[0]
    selected_peer_group = selected_cu_data['Peer_Group'].iloc[0]

    
    # Count ATMs for the selected credit union
    selected_cu_atms_count = ATM_Locations[ATM_Locations['CU_NAME'] == selected_cu_name].shape[0]

    # State average ATMs calculation
    state_atms_counts = ATM_Locations[ATM_Locations['PhysicalAddressStateCode'] == selected_state_name].groupby('CU_NAME').size()
    state_average_atms = state_atms_counts.mean()

    # Peer group ATMs calculation
    peer_cus = df_combined[df_combined['Peer_Group'] == selected_peer_group]['CU_NAME'].unique()
    peer_atms_counts = ATM_Locations[ATM_Locations['CU_NAME'].isin(peer_cus)].groupby('CU_NAME').size()
    peer_average_atms = peer_atms_counts.mean()

    # Visualization
    labels = [selected_cu_name, "State Average", "Peer Group Average"]
    values = [selected_cu_atms_count, state_average_atms, peer_average_atms]

    fig = px.bar(
        x=labels, 
        y=values, 
        labels={'x': '', 'y': 'Number of ATMs'},
        title=f"ATM Locations Comparison for {selected_cu_name}",
        text=values,
        color=labels,
        color_discrete_map={
            labels[0]: '#19618A', 
            labels[1]: '#005950', 
            labels[2]: '#09B39C'
        }
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(title='Number of ATMs'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20)
    )

    fig.update_traces(
        texttemplate='%{y}', 
        textposition='outside'
    )

    return fig

def calculate_members_per_atm(selected_cu_name, ATM_Locations, df_combined):
# Get data for the selected credit union
    selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
    selected_state_name = selected_cu_data['STATE'].iloc[0]
    selected_peer_group = selected_cu_data['Peer_Group'].iloc[0]
    selected_cu_members = selected_cu_data['NO OF MEMBERS'].iloc[0]
    
    # Count ATMs for the selected credit union
    selected_cu_atms_count = ATM_Locations[ATM_Locations['CU_NAME'] == selected_cu_name].shape[0]
    members_per_atm_cu = selected_cu_members / selected_cu_atms_count if selected_cu_atms_count else 0

    # State average calculation
    state_cus = df_combined[df_combined['STATE'] == selected_state_name]
    state_members_avg = state_cus['NO OF MEMBERS'].mean()
    state_atms_counts = ATM_Locations[ATM_Locations['PhysicalAddressStateCode'] == selected_state_name].groupby('CU_NAME').size().mean()
    members_per_atm_state = state_members_avg / state_atms_counts if state_atms_counts else 0

    # Peer group average calculation
    peer_group_cus = df_combined[df_combined['Peer_Group'] == selected_peer_group]
    peer_group_members_avg = peer_group_cus['NO OF MEMBERS'].mean()
    peer_cus_names = peer_group_cus['CU_NAME'].unique()
    peer_group_atms_counts = ATM_Locations[ATM_Locations['CU_NAME'].isin(peer_cus_names)].groupby('CU_NAME').size().mean()
    members_per_atm_peer = peer_group_members_avg / peer_group_atms_counts if peer_group_atms_counts else 0

    # Visualization
    labels = ["CIATDEL", "State Average", "Peer Group Average"]
    values = [members_per_atm_cu, members_per_atm_state, members_per_atm_peer]

    fig = px.bar(
        x=labels,
        y=values,
        labels={'x': '', 'y': 'Members per ATM'},
        title=f"Members per ATM for {selected_cu_name}",
        text=values,
        color=labels,
        color_discrete_map={
            labels[0]: '#19618A', 
            labels[1]: '#005950', 
            labels[2]: '#09B39C'
        }
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Members per ATM'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20)
    )

    fig.update_traces(
        texttemplate='%{y:.2f}', 
        textposition='outside'
    )

    return fig

def plot_members_comparison(selected_cu_name, df_combined):
    # Extract details for the selected CU
    selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name].iloc[0]
    selected_state_name = selected_cu_data['STATE']
    selected_peer_group = selected_cu_data['Peer_Group']
    
    # Total members for the selected credit union
    selected_cu_members = selected_cu_data['NO OF MEMBERS']
    
    # State average members calculation
    state_cus = df_combined[df_combined['STATE'] == selected_state_name]
    state_members_avg = state_cus['NO OF MEMBERS'].mean()
    
    # Peer group average members calculation
    peer_group_cus = df_combined[df_combined['Peer_Group'] == selected_peer_group]
    peer_group_members_avg = peer_group_cus['NO OF MEMBERS'].mean()

    # Visualization
    labels = ["CITADEL", "State Average", "Peer Group Average"]
    values = [selected_cu_members, state_members_avg, peer_group_members_avg]

    fig = px.bar(
        x=labels, 
        y=values, 
        labels={'x': '', 'y': 'Number of Members'},
        title=f"Members Comparison for {selected_cu_name}",
        text=values,
        color=labels,
        color_discrete_map={
            labels[0]: '#19618A', 
            labels[1]: '#005950', 
            labels[2]: '#09B39C'
        }
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(title='Number of Members'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=20)
    )

    fig.update_traces(
        texttemplate='%{y:.2f}', 
        textposition='outside'
    )

    return fig

def calculate_growth_rate(current_value, previous_value):
    if previous_value == 0:
        return np.nan
    else:
        return ((current_value - previous_value) / previous_value) * 100

def plot_members_growth_over_quarters(selected_cu_name, df_combined):
    """
    Plots growth rates over quarters for the selected credit union, its state, and its peer group.
    
    Parameters:
    - selected_cu_name: The name of the selected credit union.
    - df_combined: DataFrame containing credit union details.
    """
    # Extract state and peer group for the selected CU
    selected_cu_data = df_combined[df_combined['CU_NAME'] == selected_cu_name]
    selected_state = selected_cu_data['STATE'].iloc[0]
    selected_peer_group = selected_cu_data['Peer_Group'].iloc[0]
    
    # Prepare the DataFrame for growth rate calculations
    available_quarters = sorted(selected_cu_data['Quarter'].unique())
    
    # Initialize lists for storing growth rates
    growth_rate_cu, growth_rate_state, growth_rate_peer = [], [], []
    
    for i in range(1, len(available_quarters)):
        current_quarter, previous_quarter = available_quarters[i], available_quarters[i - 1]
        
        # CU growth rate
        current_value_cu = df_combined[(df_combined['CU_NAME'] == selected_cu_name) & (df_combined['Quarter'] == current_quarter)]['NO OF MEMBERS'].values[0]
        previous_value_cu = df_combined[(df_combined['CU_NAME'] == selected_cu_name) & (df_combined['Quarter'] == previous_quarter)]['NO OF MEMBERS'].values[0]
        growth_rate_cu.append(calculate_growth_rate(current_value_cu, previous_value_cu))
        
        # State growth rate
        current_value_state = df_combined[(df_combined['STATE'] == selected_state) & (df_combined['Quarter'] == current_quarter)]['NO OF MEMBERS'].mean()
        previous_value_state = df_combined[(df_combined['STATE'] == selected_state) & (df_combined['Quarter'] == previous_quarter)]['NO OF MEMBERS'].mean()
        growth_rate_state.append(calculate_growth_rate(current_value_state, previous_value_state))
        
        # Peer group growth rate
        current_value_peer = df_combined[(df_combined['Peer_Group'] == selected_peer_group) & (df_combined['Quarter'] == current_quarter)]['NO OF MEMBERS'].mean()
        previous_value_peer = df_combined[(df_combined['Peer_Group'] == selected_peer_group) & (df_combined['Quarter'] == previous_quarter)]['NO OF MEMBERS'].mean()
        growth_rate_peer.append(calculate_growth_rate(current_value_peer, previous_value_peer))

    # Plotting
    quarter_labels = [f"Q{int(q.split('-')[1]) % 4 + 1}-{q.split('-')[0][-2:]}" for q in available_quarters[1:]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_cu, mode='lines+markers', name=f'{selected_cu_name}'))
    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_state, mode='lines+markers', name='State Average'))
    fig.add_trace(go.Scatter(x=quarter_labels, y=growth_rate_peer, mode='lines+markers', name='Peer Group Average'))
    fig.data[0].name = "CU"

    fig.update_layout(
        title=f"Membership Growth Rate Over Quarters for {selected_cu_name}",
        xaxis_title="Quarter",
        yaxis_title="Growth Rate (%)",
        legend_title="Legend",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

with col1:
    st.write(" ")
    fig = generate_atm_locations_comparison_plot("CITADEL", ATM_Locations, df_combined)
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_members_comparison("CITADEL", df_combined)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.write(" ")
    
    fig = calculate_members_per_atm("CITADEL", ATM_Locations, df_combined)
    st.plotly_chart(fig, use_container_width=True)
    plot_members_growth_over_quarters("CITADEL", df_combined)

