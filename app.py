import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cluster_customer_data.csv")
        
        # Convert column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        
        # Define expected columns with possible variations
        expected_columns = {
            'gender': ['gender', 'sex'],
            'age': ['age'],
            'income': ['annual income (k$)', 'income', 'annual income'],
            'spending': ['spending score (1-100)', 'spending score', 'spendingscore']
        }
        
        # Find matching columns
        matched_cols = {}
        for col_type, possible_names in expected_columns.items():
            for name in possible_names:
                if name.lower() in df.columns:
                    matched_cols[col_type] = name.lower()
                    break
            else:
                st.error(f"Could not find {col_type} column (tried: {', '.join(possible_names)})")
                return None, None, None
        
        # Rename columns to standard names
        df = df.rename(columns={
            matched_cols['gender']: 'gender',
            matched_cols['age']: 'age',
            matched_cols['income']: 'annual_income',
            matched_cols['spending']: 'spending_score'
        })
        
        # Convert gender to numerical (Male:1, Female:0)
        df['gender'] = df['gender'].str.lower().map({'male': 1, 'female': 0})
        
        # Feature engineering
        features = df[['gender', 'age', 'annual_income', 'spending_score']]
        scaler = StandardScaler()
        df[features.columns] = scaler.fit_transform(features)
        
        # Cluster the data
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['cluster'] = kmeans.fit_predict(features)
        
        return df, features.columns.tolist(), scaler
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

df, feature_names, scaler = load_data()

# Calculate WCSS for the Elbow Method
wcss = []
if df is not None:
    features_for_elbow = df[['gender', 'age', 'annual_income', 'spending_score']]
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(features_for_elbow)
        wcss.append(kmeans.inertia_)

# Show elbow method plot
if df is not None:
    with st.expander("📈 Show Elbow Method to Determine Optimal Clusters (k)"):
        st.subheader("Elbow Method")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=list(range(1, 11)),
                y=wcss,
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(width=2),
                name='WCSS'
            )
        )
        fig_elbow.add_trace(
            go.Scatter(
                x=[5],  # Changed from optimal_k to fixed value 5
                y=[wcss[4]],  # Changed index from optimal_k-1 to 4 (since we're using k=5)
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Optimal k'
            )
        )
        fig_elbow.update_layout(
            title="Elbow Curve to Find Optimal k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="WCSS",
            showlegend=True,
            annotations=[
                dict(
                    x=5,  # Changed from optimal_k to fixed value 5
                    y=wcss[4],  # Changed index from optimal_k-1 to 4
                    xref="x",
                    yref="y",
                    text="Optimal k = 5",  # Changed from f"Optimal k = {optimal_k}"
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30
                )
            ]
        )
        st.plotly_chart(fig_elbow, use_container_width=True)


# Streamlit app
st.title("Customer Segmentation Dashboard")

with st.sidebar:
    st.header("Customer Details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 1, 100, 30)
    income = st.slider("Annual Income (k$)", 0, 200, 50)  # Adjusted range to match typical dataset
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)  # Adjusted range to match dataset
    
    submitted = st.button("Predict Cluster")

if submitted and df is not None:
    try:
        # Prepare input data
        gender_num = 1 if gender == "Male" else 0
        input_data = pd.DataFrame([[gender_num, age, income, spending]], 
                                columns=['gender', 'age', 'annual_income', 'spending_score'])
        
        # Scale the input data using the same scaler as training data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict cluster
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(df[['gender', 'age', 'annual_income', 'spending_score']])
        cluster_label = kmeans.predict(input_data_scaled)[0]
        st.success(f"Predicted Cluster: {cluster_label}")
        
        # Convert scaled values back to original scale for display
        cluster_df = df[df['cluster'] == cluster_label].copy()
        cluster_df[['gender', 'age', 'annual_income', 'spending_score']] = scaler.inverse_transform(
            cluster_df[['gender', 'age', 'annual_income', 'spending_score']]
        )
        
        st.subheader(f"Cluster {cluster_label} Characteristics")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Age", f"{round(cluster_df['age'].mean(), 1)} years")
        with col2:
            st.metric("Avg Income", f"${round(cluster_df['annual_income'].mean(), 1)}k")
        with col3:
            st.metric("Avg Spending", round(cluster_df['spending_score'].mean(), 1))
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Spending Patterns", "3D View", "Detailed Histograms"])
        
        with tab1:
            # Gender distribution
            fig_gender = px.pie(cluster_df, names='gender', 
                               title='Gender Distribution',
                               color='gender',
                               color_discrete_map={0: 'pink', 1: 'blue'},
                               labels={'0': 'Female', '1': 'Male'})
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age distribution
            fig_age = px.histogram(cluster_df, x='age', 
                                 title='Age Distribution',
                                 nbins=20,
                                 color_discrete_sequence=['purple'])
            st.plotly_chart(fig_age, use_container_width=True)
        
        with tab2:
            # Income vs Spending
            fig_spending = px.scatter(cluster_df, 
                                     x='annual_income', 
                                     y='spending_score',
                                     color='gender',
                                     title='Income vs Spending',
                                     color_discrete_map={0: 'pink', 1: 'blue'},
                                     labels={'0': 'Female', '1': 'Male'})
            st.plotly_chart(fig_spending, use_container_width=True)
            
            # Spending score distribution
            fig_score_dist = px.histogram(cluster_df, x='spending_score',
                                        title='Spending Score Distribution',
                                        nbins=20,
                                        color_discrete_sequence=['green'])
            st.plotly_chart(fig_score_dist, use_container_width=True)
        
        with tab3:
            # 3D visualization
            fig_3d = px.scatter_3d(cluster_df,
                                  x='age',
                                  y='annual_income',
                                  z='spending_score',
                                  color='gender',
                                  title='3D Customer Profile',
                                  color_discrete_map={0: 'pink', 1: 'blue'},
                                  labels={'0': 'Female', '1': 'Male'})
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab4:
            # Create subplots for all histograms
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=("Age Distribution", "Annual Income Distribution", 
                                              "Spending Score Distribution", "Gender Distribution"))
            
            # Age histogram
            fig.add_trace(
                go.Histogram(x=cluster_df['age'], nbinsx=20, marker_color='purple'),
                row=1, col=1
            )
            
            # Income histogram
            fig.add_trace(
                go.Histogram(x=cluster_df['annual_income'], nbinsx=20, marker_color='blue'),
                row=1, col=2
            )
            
            # Spending score histogram
            fig.add_trace(
                go.Histogram(x=cluster_df['spending_score'], nbinsx=20, marker_color='green'),
                row=2, col=1
            )
            
            # Gender histogram
            gender_counts = cluster_df['gender'].value_counts()
            fig.add_trace(
                go.Bar(x=['Female', 'Male'], y=gender_counts, marker_color=['pink', 'lightblue']),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(height=800, width=800, showlegend=False, title_text="Cluster Characteristics Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
elif submitted:
    st.error("System not ready. Please check if data loaded properly.")

# Add overall data visualization section
if df is not None:
    # Convert scaled values back to original scale for visualization
    df_display = df.copy()
    df_display[['gender', 'age', 'annual_income', 'spending_score']] = scaler.inverse_transform(
        df[['gender', 'age', 'annual_income', 'spending_score']]
    )
    
    st.header("Overall Data Visualization")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Feature Distributions", "Cluster Comparison"])
    
    with viz_tab1:
        # Feature distribution across all data
        st.subheader("Feature Distributions Across All Data")
        
        # Create a 2x2 grid of histograms
        fig_all = make_subplots(rows=2, cols=2, 
                              subplot_titles=("Age Distribution (All)", "Income Distribution (All)", 
                                            "Spending Score Distribution (All)", "Gender Distribution (All)"))
        
        # Age histogram
        fig_all.add_trace(
            go.Histogram(x=df_display['age'], nbinsx=20, marker_color='purple'),
            row=1, col=1
        )
        
        # Income histogram
        fig_all.add_trace(
            go.Histogram(x=df_display['annual_income'], nbinsx=20, marker_color='blue'),
            row=1, col=2
        )
        
        # Spending score histogram
        fig_all.add_trace(
            go.Histogram(x=df_display['spending_score'], nbinsx=20, marker_color='green'),
            row=2, col=1
        )
        
        # Gender histogram
        gender_counts_all = df_display['gender'].value_counts()
        fig_all.add_trace(
            go.Bar(x=['Female', 'Male'], y=gender_counts_all, marker_color=['pink', 'lightblue']),
            row=2, col=2
        )
        
        fig_all.update_layout(height=800, width=800, showlegend=False)
        st.plotly_chart(fig_all, use_container_width=True)
    
    with viz_tab2:
        # Cluster comparison visualizations
        st.subheader("Cluster Comparison")
        
        # Box plots comparing clusters
        fig_boxes = make_subplots(rows=1, cols=3, 
                                 subplot_titles=("Age by Cluster", "Income by Cluster", " Ros Spending Score by Cluster"))
        
        # Age box plot
        fig_boxes.add_trace(
            go.Box(x=df_display['cluster'], y=df_display['age'], name='Age', marker_color='purple'),
            row=1, col=1
        )
        
        # Income box plot
        fig_boxes.add_trace(
            go.Box(x=df_display['cluster'], y=df_display['annual_income'], name='Income', marker_color='blue'),
            row=1, col=2
        )
        
        # Spending score box plot
        fig_boxes.add_trace(
            go.Box(x=df_display['cluster'], y=df_display['spending_score'], name='Spending Score', marker_color='green'),
            row=1, col=3
        )
        
        fig_boxes.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_boxes, use_container_width=True)
        
        # Cluster sizes
        cluster_sizes = df_display['cluster'].value_counts().sort_index()
        fig_cluster_sizes = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                                 labels={'x': 'Cluster', 'y': 'Number of Customers'},
                                 title='Number of Customers in Each Cluster',
                                 color=cluster_sizes.index,
                                 color_continuous_scale='Viridis')
        st.plotly_chart(fig_cluster_sizes, use_container_width=True)