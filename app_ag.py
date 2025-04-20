"""
Insurance Fraud Detection Streamlit Application with Plotly Visualizations
and AgGrid for enhanced table display

This application demonstrates using graph databases and machine learning
to detect potentially fraudulent insurance claims.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from gqlalchemy import Memgraph
import os
import pickle
import hashlib
import time

# Import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Import modules
from model import (
    train_fraud_detection_model, 
    predict_fraud, 
    extract_features,
    prepare_new_claim_features,
    FeatureStore
)

# Configure page
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Set up database connection
def get_db_connection():
    """Connect to Memgraph database"""
    try:
        return Memgraph(host="localhost", port=7687)
    except Exception as e:
        st.error(f"Failed to connect to Memgraph: {e}")
        st.error("Make sure Memgraph is running and accessible at localhost:7687")
        st.info("You can start Memgraph using: docker run -it -p 7687:7687 -p 3000:3000 memgraph/memgraph-platform")
        return None

# Configure AgGrid
def configure_aggrid(df, selection_mode='single', height=400, fit_columns=True, key=None, pagination=True, page_size=20):
    """Configure AgGrid with consistent settings"""
    # Check if DataFrame is empty or None
    if df is None or (hasattr(df, 'empty') and df.empty):
        # Return a default empty grid
        empty_df = pd.DataFrame({"No Data": ["No data available"]})
        gb = GridOptionsBuilder.from_dataframe(empty_df)
        grid_options = gb.build()
        return AgGrid(
            empty_df,
            gridOptions=grid_options,
            height=height,
            key=key
        )
    
    # Normal GridOptionsBuilder for non-empty DataFrame
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Add selection
    if selection_mode:
        gb.configure_selection(selection_mode=selection_mode, use_checkbox=False)
    
    # Auto-size columns to fit content
    if fit_columns:
        gb.configure_auto_height(False)
        gb.configure_grid_options(domLayout='autoHeight')
    
    # Configure pagination
    if pagination:
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=page_size)
    
    # Configure columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=2)
        elif col == 'is_fraud':
            gb.configure_column(
                col, 
                type=["booleanColumn"],
                cellRenderer=JsCode('''
                function(params) {
                    if (params.value === true) {
                        return '<span style="color: red; font-weight: bold;">⚠️ Yes</span>';
                    } else if (params.value === false) {
                        return '<span style="color: green;">No</span>';
                    } else {
                        return '<span style="color: gray;">Unknown</span>';
                    }
                }
                ''')
            )
    
    # Create grid options
    grid_options = gb.build()
    
    # Create AgGrid
    return AgGrid(
        df,
        gridOptions=grid_options,
        height=height,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=fit_columns,
        key=key
    )

# Get claim details
def get_claim_details(db, claim_id):
    """Get detailed information about a claim"""
    try:
        claim_info = db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[:ON_INCIDENT]->(incident:INCIDENT)
            RETURN claim, incident
        """)
        
        individuals = db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[:ON_INCIDENT]->(incident:INCIDENT)-[:INCIDENT]->(individual:INDIVIDUAL)
            RETURN individual.first_name AS first_name, individual.last_name AS last_name
        """)
        
        payments = db.execute_and_fetch(f"""
            MATCH (payment:CLAIM_PAYMENT)-[:ON_CLAIM]->(claim:CLAIM {{clm_id: '{claim_id}'}})
            RETURN payment.amount AS amount, payment.pay_id AS payment_id
        """)
        
        return {
            "claim_info": list(claim_info) if claim_info else [],
            "individuals": list(individuals) if individuals else [],
            "payments": list(payments) if payments else []
        }
    except Exception as e:
        st.error(f"Error fetching claim details: {e}")
        return {
            "claim_info": [],
            "individuals": [],
            "payments": []
        }

# Model caching functions
def should_retrain_model(db):
    """Check if model should be retrained by comparing database state"""
    try:
        # Get counts of key entities to represent DB state
        claim_count = list(db.execute_and_fetch("MATCH (c:CLAIM) RETURN count(c) as count"))[0]["count"]
        fraud_count = list(db.execute_and_fetch("MATCH (c:CLAIM {fraud: true}) RETURN count(c) as count"))[0]["count"]
        
        # Create a simple state signature
        current_state = f"{claim_count}_{fraud_count}_{int(time.time() / 3600)}"  # Changes at most once per hour
        state_hash = hashlib.md5(current_state.encode()).hexdigest()
        
        # Check if model exists and state file matches
        if os.path.exists('model_cache/model.pkl') and os.path.exists('model_cache/state.txt'):
            with open('model_cache/state.txt', 'r') as f:
                saved_hash = f.read().strip()
                if saved_hash == state_hash:
                    return False  # No need to retrain
        
        # Save the current state hash
        os.makedirs('model_cache', exist_ok=True)
        with open('model_cache/state.txt', 'w') as f:
            f.write(state_hash)
        
        return True  # Need to retrain
    except Exception as e:
        print(f"Error checking model status: {e}")
        return True  # Retrain if there's any error

def save_model(model_data):
    """Save model data to disk"""
    try:
        os.makedirs('model_cache', exist_ok=True)
        with open('model_cache/model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model():
    """Load model data from disk"""
    try:
        with open('model_cache/model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Safe access to DataFrames and collections
def safe_get_first_item(collection):
    """Safely get the first item from a collection or return None"""
    if collection is None:
        return None
    if not hasattr(collection, '__len__'):
        return None
    if len(collection) == 0:
        return None
    return collection[0]

def is_empty(collection):
    """Check if a collection is None or empty"""
    if collection is None:
        return True
    if hasattr(collection, 'empty') and collection.empty:
        return True
    if hasattr(collection, '__len__') and len(collection) == 0:
        return True
    return False

def has_items(collection):
    """Check if a collection has any items (safe for any type)"""
    if collection is None:
        return False
    if hasattr(collection, 'empty'):
        return not collection.empty
    if hasattr(collection, '__len__'):
        return len(collection) > 0
    return False

# Display overview page
def display_overview(db):
    """Display dashboard with statistics and system overview"""
    st.header("Insurance Fraud Detection Overview")
    
    st.write("""
    This application demonstrates how graph databases can be used for detecting fraudulent insurance claims.
    The system uses Memgraph to store insurance data and applies machine learning to identify potential fraud.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Statistics")
        try:
            claim_count = list(db.execute_and_fetch("MATCH (c:CLAIM) RETURN count(c) as count"))[0]["count"]
            fraud_count = list(db.execute_and_fetch("MATCH (c:CLAIM {fraud: true}) RETURN count(c) as count"))[0]["count"]
            individual_count = list(db.execute_and_fetch("MATCH (i:INDIVIDUAL) RETURN count(i) as count"))[0]["count"]
            policy_count = list(db.execute_and_fetch("MATCH (p:POLICY) RETURN count(p) as count"))[0]["count"]
            
            st.metric("Total Claims", claim_count)
            st.metric("Fraudulent Claims", fraud_count)
            st.metric("Individuals", individual_count)
            st.metric("Policies", policy_count)
        except Exception as e:
            st.error(f"Error fetching database statistics: {e}")
    
    with col2:
        st.subheader("Fraud Percentage")
        try:
            if claim_count > 0:
                fraud_percentage = (fraud_count / claim_count) * 100
                
                # Create a Plotly pie chart
                fig = px.pie(
                    values=[fraud_count, claim_count - fraud_count],
                    names=['Fraud', 'Legitimate'],
                    color_discrete_sequence=['#ff9999', '#66b3ff'],
                    title="Fraud vs Legitimate Claims",
                    height=300,
                    width=300
                )
                
                # Customize layout for a smaller chart
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(
                        font=dict(size=10),
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5
                    ),
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("No claims data available")
        except Exception as e:
            st.error(f"Error creating fraud percentage chart: {e}")
    
    st.subheader("Key Features for Fraud Detection")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        ### Network Analysis
        
        The graph structure allows us to identify connections between:
        - Individuals involved in multiple claims
        - Claims filed against the same policies
        - Incidents occurring at the same addresses
        
        These connections can reveal suspicious patterns that might indicate fraud rings.
        """)
        
        st.markdown("""
        ### Feature Extraction
        
        We extract several features from the graph:
        - Number of individuals involved in a claim
        - Total amount paid for a claim
        - Whether the policy has expired
        - Premium type of the policy
        """)
    
    with feature_col2:
        st.markdown("""
        ### Community Detection
        
        Graph algorithms can identify communities within the data:
        - Groups of claims, individuals, and policies that are closely connected
        - Communities with high fraud rates are flagged for investigation
        - New claims associated with these communities receive higher scrutiny
        """)
        
        st.markdown("""
        ### Machine Learning
        
        We combine graph features with traditional data to train fraud detection models:
        - Logistic regression identifies suspicious patterns
        - Upsampling balances the dataset for better training
        - Model performance is evaluated with precision, recall, and F1 score
        """)

# Display claims exploration page
def explore_claims(db):
    """Display claims exploration page"""
    st.header("Explore Insurance Claims")
    
    try:
        # Get claims for display
        claims_query = db.execute_and_fetch("""
            MATCH (c:CLAIM)
            RETURN c.clm_id AS claim_id, 
                   c.amount AS amount, 
                   c.fraud AS is_fraud,
                   c.status AS status
            LIMIT 100
        """)
        
        # Safe conversion to list with error handling
        try:
            claims = list(claims_query) if claims_query else []
        except:
            claims = []
            
        if not claims:
            st.info("No claims found in the database")
            return
            
        # Convert to DataFrame for display
        claims_df = pd.DataFrame(claims)
        
        # Add filters
        st.subheader("Filter Claims")
        col1, col2 = st.columns(2)
        
        with col1:
            fraud_filter = st.selectbox(
                "Filter by fraud status:",
                options=["All", "Fraudulent", "Legitimate"]
            )
        
        with col2:
            status_values = ["All"]
            if "status" in claims_df.columns:
                for val in claims_df["status"].dropna().unique():
                    if val is not None:
                        status_values.append(val)
                
            status_filter = st.selectbox(
                "Filter by claim status:",
                options=status_values
            )
        
        # Apply filters
        filtered_df = claims_df.copy()
        
        if fraud_filter == "Fraudulent":
            filtered_df = filtered_df[filtered_df["is_fraud"] == True]
        elif fraud_filter == "Legitimate":
            filtered_df = filtered_df[filtered_df["is_fraud"] == False]
            
        if status_filter != "All" and "status" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["status"] == status_filter]
        
        # Display claims table using AgGrid
        st.subheader("Claims")
        
        if hasattr(filtered_df, 'empty') and filtered_df.empty:
            st.info("No claims match the filter criteria")
            return
            
        # Use AgGrid for the claims table
        response = configure_aggrid(
            filtered_df,
            selection_mode='single',
            height=400,
            pagination=True,
            page_size=20,
            key="claims_grid"
        )
        
        # Get selected row from AgGrid
        selected_rows = response.get("selected_rows")
        
        # Check if selected_rows is None or empty
        if selected_rows is None or (hasattr(selected_rows, '__len__') and len(selected_rows) == 0):
            st.info("Select a claim from the table to view details")
            return
            
        # Get the selected claim ID - handle different return types
        selected_claim = None
        try:
            # Try accessing as a list of dictionaries
            if isinstance(selected_rows, list) and len(selected_rows) > 0:
                selected_claim = selected_rows[0].get("claim_id")
            # Try accessing as a DataFrame
            elif hasattr(selected_rows, 'iloc') and len(selected_rows) > 0:
                selected_claim = selected_rows.iloc[0].get("claim_id")
        except Exception as e:
            st.error(f"Error accessing selected claim: {e}")
            return
            
        if not selected_claim:
            st.error("Could not determine selected claim. Please try selecting again.")
            return
        
        st.markdown("---")
        st.subheader(f"Details for Claim: {selected_claim}")
        
        # Get detailed information about the claim
        details = get_claim_details(db, selected_claim)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Claim Information")
            claim_info_list = details.get("claim_info", [])
            if claim_info_list and len(claim_info_list) > 0:
                claim_data = claim_info_list[0]
                claim = claim_data.get("claim") if isinstance(claim_data, dict) else None
                incident = claim_data.get("incident") if isinstance(claim_data, dict) else None
                
                if claim and incident:
                    st.markdown(f"**Claim ID:** {getattr(claim, 'clm_id', 'Unknown')}")
                    st.markdown(f"**Amount:** {getattr(claim, 'amount', 'Unknown')}")
                    st.markdown(f"**Fraud:** {getattr(claim, 'fraud', 'Unknown')}")
                    st.markdown(f"**Status:** {getattr(claim, 'status', 'Unknown')}")
                    st.markdown(f"**Incident ID:** {getattr(incident, 'inc_id', 'Unknown')}")
                    st.markdown(f"**Incident Date:** {getattr(incident, 'accident_date', 'Unknown')}")
                else:
                    st.info("Incomplete claim or incident information")
            else:
                st.info("No detailed information found for this claim")
        
        with col2:
            st.subheader("Involved Individuals")
            individuals_list = details.get("individuals", [])
            if individuals_list and len(individuals_list) > 0:
                individuals_df = pd.DataFrame(individuals_list)
                # Use AgGrid for individuals
                configure_aggrid(
                    individuals_df, 
                    selection_mode=None, 
                    height=200,
                    pagination=True,
                    page_size=20,
                    key="individuals_grid"
                )
            else:
                st.info("No individuals associated with this claim")
        
        st.subheader("Payments")
        payments_list = details.get("payments", [])
        if payments_list and len(payments_list) > 0:
            payments_df = pd.DataFrame(payments_list)
            # Use AgGrid for payments
            configure_aggrid(
                payments_df, 
                selection_mode=None, 
                height=200,
                pagination=True,
                page_size=20,
                key="payments_grid"
            )
            
            # Create a payment chart with Plotly
            if hasattr(payments_df, 'empty') and not payments_df.empty and "amount" in payments_df.columns:
                st.subheader("Payment Distribution")
                
                fig = px.bar(
                    payments_df, 
                    x="payment_id", 
                    y="amount",
                    title="Payments for this Claim",
                    labels={"payment_id": "Payment ID", "amount": "Amount ($)"},
                    color_discrete_sequence=['#2a9df4']
                )
                
                fig.update_layout(
                    xaxis_title="Payment ID",
                    yaxis_title="Amount ($)",
                    height=400
                )
                
                st.plotly_chart(fig)
        else:
            st.info("No payments associated with this claim")
        
        # Display graph visualization query
        st.subheader("Graph Query for Visualization")
        query = f"""
        MATCH p=(claim:CLAIM {{clm_id: '{selected_claim}'}})-[*1..2]-()
        RETURN p
        """
        st.code(query, language="cypher")
        st.info("Run this query in Memgraph Lab or another graph visualization tool to see the connections")
        
        # Extract features for this claim
        st.subheader("Extracted Features")
        try:
            # Get the features
            features = extract_features(db, selected_claim)
            
            # Create a DataFrame with the features
            features_df = pd.DataFrame([features]) if features else pd.DataFrame()
            
            if hasattr(features_df, 'empty') and not features_df.empty:
                # Handle categorical feature
                premium_col = FeatureStore.POL_PREMIUM
                if premium_col in features_df.columns:
                    premium = features_df[premium_col][0]
                    for p_type in ["A", "B", "C", "D", "U"]:
                        features_df[f"policy_premium_{p_type}"] = 1 if p_type == premium else 0
                    features_df = features_df.drop(columns=[premium_col])
                else:
                    # Handle case when premium column is missing
                    for p_type in ["A", "B", "C", "D", "U"]:
                        features_df[f"policy_premium_{p_type}"] = 0
                    # Set a default
                    features_df["policy_premium_U"] = 1
                
                # Use AgGrid for features
                configure_aggrid(
                    features_df, 
                    selection_mode=None, 
                    height=150,
                    pagination=True,
                    page_size=20,
                    key="features_grid"
                )
                
                # Visualize feature values with Plotly
                numeric_features = [
                    FeatureStore.IND_COUNT,
                    FeatureStore.AMT_PAID,
                    FeatureStore.NUM_FRAUDS_COMMUNITY,
                    FeatureStore.NUM_FRAUDS_NEIGHBORHOOD,
                    FeatureStore.INFLUENCE
                ]
                
                # Make sure all features exist in the DataFrame
                valid_features = [f for f in numeric_features if f in features_df.columns]
                
                if valid_features and len(valid_features) > 0:
                    # Plot numeric features as a horizontal bar chart
                    feature_values = features_df[valid_features].iloc[0]
                    
                    fig = px.bar(
                        x=feature_values.values,
                        y=feature_values.index,
                        orientation='h',
                        title="Numeric Feature Values",
                        labels={"x": "Value", "y": "Feature"},
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        height=300
                    )
                    
                    st.plotly_chart(fig)
                
                # Show feature descriptions
                with st.expander("Feature Descriptions"):
                    st.markdown("""
                    - **ind_count**: Number of individuals involved in the claim
                    - **amount_paid**: Total amount paid for the claim
                    - **num_frauds_community**: Number of fraudulent claims in the same community
                    - **num_frauds_neighborhood**: Number of fraudulent claims within 4 hops in the graph
                    - **influence**: PageRank score in the graph (centrality measure)
                    - **policy_premium_X**: One-hot encoded policy premium type (A, B, C, D, or Unknown)
                    """)
            else:
                st.warning("No features extracted for this claim")
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            st.exception(e)  # This will show the full traceback for debugging
                
    except Exception as e:
        st.error(f"Error exploring claims: {e}")
        st.exception(e)  # This will show the full traceback for debugging

# Function to analyze claims
def analyze_claim(db, model_data, claim_id):
    """Analyze a claim for fraud detection"""
    # Extract features
    try:
        features = extract_features(db, claim_id)
        
        # Create a DataFrame with the features
        features_df = pd.DataFrame([features]) if features else pd.DataFrame()
        
        if features_df is None or (hasattr(features_df, 'empty') and features_df.empty):
            st.error("Could not extract features for this claim")
            return None, None
        
        # Handle categorical feature
        premium_col = FeatureStore.POL_PREMIUM
        if premium_col in features_df.columns:
            premium = features_df[premium_col][0]
            for p_type in ["A", "B", "C", "D", "U"]:
                features_df[f"policy_premium_{p_type}"] = 1 if p_type == premium else 0
            features_df = features_df.drop(columns=[premium_col])
        else:
            # Handle missing premium column
            for p_type in ["A", "B", "C", "D", "U"]:
                features_df[f"policy_premium_{p_type}"] = 0
            # Set default
            features_df["policy_premium_U"] = 1
        
        # Make sure all expected columns are present
        feature_columns = model_data.get("feature_columns", [])
        if feature_columns is not None and hasattr(feature_columns, '__len__') and len(feature_columns) > 0:
            # Loop through expected columns and add missing ones
            # Convert feature_columns to list if it's an Index
            if hasattr(feature_columns, 'tolist'):
                feature_list = feature_columns.tolist()
            else:
                feature_list = list(feature_columns)
                
            for col in feature_list:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Keep only required columns in the same order
            # Filter to only include columns that exist in DataFrame
            columns_to_use = [col for col in feature_list if col in features_df.columns]
            if columns_to_use and len(columns_to_use) > 0:
                features_df = features_df[columns_to_use]
        
        # Make prediction
        prediction = predict_fraud(model_data, features_df)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction.get("is_fraud", False):
                st.error("⚠️ Potential Fraud Detected!")
            else:
                st.success("✓ Claim appears legitimate")
        
        with col2:
            fraud_prob = prediction.get("fraud_probability", 0) * 100
            st.metric("Fraud Probability", f"{fraud_prob:.1f}%")
        
        # Place the gauge and risk factors side by side
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            # Create a gauge chart with Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fraud_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Risk"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if fraud_prob > 75 else "orange" if fraud_prob > 50 else "yellow" if fraud_prob > 25 else "green"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "lightyellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig)
        
        with vis_col2:
            # Show feature importance
            st.subheader("Key Risk Factors")
            
            # Check if feature_columns and model exist
            model = model_data.get("model")
            feature_columns_exist = feature_columns is not None and hasattr(feature_columns, '__len__') and len(feature_columns) > 0
            
            if model is not None and feature_columns_exist and hasattr(model, "coef_"):
                # Get top features based on model coefficients
                coefs = model.coef_[0]
                
                # Create feature importance DataFrame
                # Ensure feature_columns is a list, not an Index
                if hasattr(feature_columns, 'tolist'):
                    feature_list = feature_columns.tolist()
                else:
                    feature_list = list(feature_columns)
                    
                # Make sure we have the right number of features and coefficients
                if len(feature_list) == len(coefs):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_list,
                        'Importance': coefs
                    }).sort_values('Importance', ascending=True)  # Ascending for horizontal bar chart
                    
                    # Keep only top 10 features
                    if len(feature_importance) > 10:
                        # Use iloc since head/tail return new DataFrames
                        neg_features = feature_importance.iloc[:5]  # Top 5 negative
                        pos_features = feature_importance.iloc[-5:]  # Top 5 positive
                        top_features = pd.concat([neg_features, pos_features])
                    else:
                        top_features = feature_importance
                    
                    # Create color mapping
                    colors = ['green' if x < 0 else 'red' for x in top_features['Importance']]
                    
                    # Create horizontal bar chart with Plotly
                    fig = px.bar(
                        top_features,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color_discrete_sequence=['red']
                    )
                    
                    # Update colors based on coefficient sign
                    fig.update_traces(marker_color=colors)
                    
                    # Add a vertical line at x=0
                    fig.add_shape(
                        type="line",
                        x0=0, y0=-0.5,
                        x1=0, y1=len(top_features)-0.5,
                        line=dict(color="black", width=1, dash="dash")
                    )
                    
                    fig.update_layout(
                        xaxis_title="Impact on Fraud Probability",
                        yaxis_title="Feature",
                        height=400
                    )
                    
                    st.plotly_chart(fig)
                else:
                    st.warning(f"Feature columns ({len(feature_list)}) and coefficients ({len(coefs)}) length mismatch.")
            else:
                st.warning("Model information is incomplete. Cannot display risk factors.")
        
        # Explain feature values for this claim
        st.subheader("Claim Features")
        
        # Add feature descriptions and values side by side
        feature_desc = {
            'ind_count': 'Number of individuals involved',
            'amount_paid': 'Total amount paid ($)',
            'num_frauds_community': 'Frauds in same community',
            'num_frauds_neighborhood': 'Frauds in neighborhood (within 4 hops)',
            'influence': 'PageRank centrality score',
            'policy_premium_A': 'Premium type A',
            'policy_premium_B': 'Premium type B',
            'policy_premium_C': 'Premium type C',
            'policy_premium_D': 'Premium type D',
            'policy_premium_U': 'Premium type unknown'
        }
        
        feature_values = []
        for col in features_df.columns:
            if col in feature_desc:
                value = features_df[col].values[0]
                feature_values.append({
                    'Feature': feature_desc[col],
                    'Value': value
                })
        
        # Display features table with AgGrid
        if feature_values and len(feature_values) > 0:
            feature_df = pd.DataFrame(feature_values)
            configure_aggrid(
                feature_df, 
                selection_mode=None, 
                height=300,
                pagination=True,
                page_size=20,
                key="feature_values_grid"
            )
        
        return features_df, prediction
    except Exception as e:
        st.error(f"Error analyzing claim: {e}")
        st.exception(e)
        return None, None

# Display fraud detection page
def fraud_detection(db):
    """Display fraud detection page with prediction functionality"""
    st.header("Fraud Detection")
    
    # Get or train the model
    if should_retrain_model(db):
        with st.spinner("Training fraud detection model..."):
            try:
                model_data = train_fraud_detection_model(db)
                save_model(model_data)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error training model: {e}")
                st.error("Please check the database connection and try again.")
                return
    else:
        # Load model from cache
        model_data = load_model()
        if model_data is None:
            with st.spinner("Training fraud detection model..."):
                try:
                    model_data = train_fraud_detection_model(db)
                    save_model(model_data)
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return
    
    # Create tabs for different prediction methods
    tab1, tab2 = st.tabs(["Analyze Existing Claim", "Input New Claim"])
    
    # Tab 1: Analyze existing claim
    with tab1:
        st.subheader("Analyze Existing Claim")
        
        try:
            # Get claims for display
            claims_query = db.execute_and_fetch("""
                MATCH (c:CLAIM)
                RETURN c.clm_id AS claim_id, 
                       c.amount AS amount,
                       c.fraud AS is_fraud
                LIMIT 100
            """)
            
            # Safe conversion to list with error handling
            try:
                claims = list(claims_query) if claims_query else []
            except:
                claims = []
            
            if not claims:
                st.info("No claims found in database")
                return
                
            # Convert to DataFrame for display
            claims_df = pd.DataFrame(claims)
            
            # Add a filter for known fraud status
            show_known_frauds = st.checkbox("Show known fraudulent claims", value=False)
            if show_known_frauds:
                filtered_claims = claims_df
            else:
                # Hide claims with known fraud status if the checkbox is not checked
                filtered_claims = claims_df[claims_df['is_fraud'].isna()]
            
            # Check if filtered DataFrame is empty
            if hasattr(filtered_claims, 'empty') and filtered_claims.empty:
                st.info("No claims match the filter criteria")
                return
                
            # Display the table of claims using AgGrid
            response = configure_aggrid(
                filtered_claims,
                selection_mode='single',
                height=400,
                pagination=True,
                page_size=20,
                key="fraud_detection_grid"
            )
            
            # Get selected row from AgGrid
            selected_rows = response.get("selected_rows")
            
            # Check if selected_rows is None or empty
            if selected_rows is None or (hasattr(selected_rows, '__len__') and len(selected_rows) == 0):
                st.info("Select a claim from the table to analyze")
                return
            
            # Get the selected claim ID - handle different return types
            selected_claim = None
            try:
                # Try accessing as a list of dictionaries
                if isinstance(selected_rows, list) and len(selected_rows) > 0:
                    selected_claim = selected_rows[0].get("claim_id")
                # Try accessing as a DataFrame
                elif hasattr(selected_rows, 'iloc') and len(selected_rows) > 0:
                    selected_claim = selected_rows.iloc[0].get("claim_id")
            except Exception as e:
                st.error(f"Error accessing selected claim: {e}")
                return
                
            if not selected_claim:
                st.error("Could not determine selected claim. Please try selecting again.")
                return
                
            # Add an analyze button
            analyze_button = st.button("Analyze Selected Claim")
            
            if analyze_button:
                # Analyze the claim
                analyze_claim(db, model_data, selected_claim)
                
        except Exception as e:
            st.error(f"Error in claim analysis: {e}")
            st.exception(e)  # This will show the full traceback for debugging
    
    # Tab 2: Input new claim
    with tab2:
        st.subheader("Input Claim Data Manually")
        
        with st.form("new_claim_form"):
            st.subheader("New Claim Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ind_count = st.number_input("Number of individuals involved", min_value=0, max_value=10, value=1)
                amount_paid = st.number_input("Amount paid ($)", min_value=0.0, max_value=100000.0, value=5000.0)
                premium_type = st.selectbox("Policy premium type", ["A", "B", "C", "D", "U"])
            
            with col2:
                num_frauds_community = st.number_input("Frauds in community", min_value=0, max_value=10, value=0)
                num_frauds_neighborhood = st.number_input("Frauds in neighborhood", min_value=0, max_value=10, value=0)
                influence = st.number_input("Influence score", min_value=0.0, max_value=1.0, value=0.001, format="%.5f")
            
            # Add descriptions
            with st.expander("Feature Descriptions"):
                st.markdown("""
                - **Number of individuals involved**: Count of people associated with the claim
                - **Amount paid**: Total amount paid for the claim
                - **Policy premium type**: Premium category (A = highest, D = lowest, U = unknown)
                - **Frauds in community**: Count of fraudulent claims in the same community
                - **Frauds in neighborhood**: Count of fraudulent claims within 4 hops in the graph
                - **Influence score**: PageRank centrality (how connected the claim is in the network)
                """)
            
            submitted = st.form_submit_button("Analyze Claim")
            
            if submitted:
                # Create manual features dictionary
                manual_features = {
                    "ind_count": ind_count,
                    "amount_paid": amount_paid,
                    "premium_type": premium_type,
                    "num_frauds_community": num_frauds_community,
                    "num_frauds_neighborhood": num_frauds_neighborhood,
                    "influence": influence
                }
                
                # Create features DataFrame
                try:
                    features = prepare_new_claim_features(manual_features)
                    
                    # Get feature columns
                    feature_columns = model_data.get("feature_columns", [])
                    feature_columns_exist = feature_columns is not None and hasattr(feature_columns, '__len__') and len(feature_columns) > 0
                    
                    # Ensure all required columns are present
                    if feature_columns_exist:
                        # Convert feature_columns to list if it's an Index
                        if hasattr(feature_columns, 'tolist'):
                            feature_list = feature_columns.tolist()
                        else:
                            feature_list = list(feature_columns)
                            
                        # Add missing columns with default values
                        for col in feature_list:
                            if col not in features.columns:
                                features[col] = 0
                        
                        # Keep only required columns in the same order
                        # Filter to include only columns that exist in features
                        columns_to_use = [col for col in feature_list if col in features.columns]
                        if columns_to_use and len(columns_to_use) > 0:
                            features = features[columns_to_use]
                    
                    # Make prediction
                    prediction = predict_fraud(model_data, features)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Fraud Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction.get("is_fraud", False):
                            st.error("⚠️ Potential Fraud Detected!")
                        else:
                            st.success("✓ Claim appears legitimate")
                    
                    with col2:
                        fraud_prob = prediction.get("fraud_probability", 0) * 100
                        st.metric("Fraud Probability", f"{fraud_prob:.1f}%")
                    
                    # Place gauge and recommendation side by side
                    vis_col1, vis_col2 = st.columns(2)
                    
                    with vis_col1:
                        # Create a gauge chart with Plotly
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = fraud_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if fraud_prob > 75 else "orange" if fraud_prob > 50 else "yellow" if fraud_prob > 25 else "green"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "lightyellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 75
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig)
                    
                    with vis_col2:
                        # Add recommendations with matching height
                        st.subheader("Next Steps")
                        
                        # Recommendation box with height to match gauge
                        recommendation_style = """
                        <style>
                        .recommendation-box {
                            height: 290px;
                            padding: 20px;
                            border-radius: 5px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        }
                        .high-risk { background-color: rgba(255, 99, 71, 0.2); }
                        .medium-risk { background-color: rgba(255, 165, 0, 0.2); }
                        .low-med-risk { background-color: rgba(255, 255, 0, 0.1); }
                        .low-risk { background-color: rgba(144, 238, 144, 0.2); }
                        </style>
                        """
                        
                        st.markdown(recommendation_style, unsafe_allow_html=True)
                        
                        if fraud_prob > 75:
                            st.markdown("""
                            <div class="recommendation-box high-risk">
                            <h3>⚠️ High Risk Action Required:</h3>
                            <ul>
                              <li>Immediately escalate to fraud investigation team</li>
                              <li>Place a hold on all claim processing</li>
                              <li>Verify all documentation thoroughly</li>
                              <li>Schedule interviews with all involved parties</li>
                              <li>Check for connections to known fraud patterns</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fraud_prob > 50:
                            st.markdown("""
                            <div class="recommendation-box medium-risk">
                            <h3>⚠️ Medium-High Risk Action Required:</h3>
                            <ul>
                              <li>Request additional verification before processing</li>
                              <li>Verify identity of all involved individuals</li>
                              <li>Cross-check with previous claims history</li>
                              <li>Conduct additional phone interviews</li>
                              <li>Document all unusual aspects of the claim</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fraud_prob > 25:
                            st.markdown("""
                            <div class="recommendation-box low-med-risk">
                            <h3>ℹ️ Medium-Low Risk Action Required:</h3>
                            <ul>
                              <li>Proceed with standard verification steps</li>
                              <li>Verify documentation is complete</li>
                              <li>Follow normal processing procedures</li>
                              <li>Note any unusual patterns for future reference</li>
                              <li>No special handling required</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="recommendation-box low-risk">
                            <h3>✓ Low Risk Action Required:</h3>
                            <ul>
                              <li>Proceed with standard processing</li>
                              <li>No additional verification needed</li>
                              <li>Process claim according to normal timeline</li>
                              <li>Apply standard documentation procedures</li>
                              <li>No special attention required</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Create radar chart of input features below the gauges
                    radar_data = {
                        'Feature': ['Individuals', 'Amount Paid', 'Community Frauds', 'Neighborhood Frauds', 'Influence'],
                        'Value': [ind_count, amount_paid/10000, num_frauds_community, num_frauds_neighborhood, influence*1000]  # Scaled for visibility
                    }
                    
                    # Create a radar chart
                    fig = px.line_polar(
                        pd.DataFrame(radar_data), 
                        r='Value', 
                        theta='Feature', 
                        line_close=True,
                        range_r=[0, max(radar_data['Value'])*1.2],
                        title="Feature Radar Chart"
                    )
                    
                    fig.update_traces(fill='toself')
                    
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error processing claim: {e}")
                    st.exception(e)

# Display model evaluation page
def model_evaluation(db):
    """Display model evaluation page with metrics and visualizations"""
    st.header("Fraud Detection Model Evaluation")
    
    # Get or train the model using caching
    if should_retrain_model(db):
        with st.spinner("Evaluating fraud detection model..."):
            try:
                model_data = train_fraud_detection_model(db)
                save_model(model_data)
            except Exception as e:
                st.error(f"Error training model: {e}")
                return
    else:
        # Load model from cache
        model_data = load_model()
        if model_data is None:
            with st.spinner("Training fraud detection model..."):
                try:
                    model_data = train_fraud_detection_model(db)
                    save_model(model_data)
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return
    
    # Check if model data is complete
    if not model_data or not all(k in model_data for k in ["y_test", "y_pred", "model"]):
        st.error("Model data is incomplete. Please retrain the model.")
        return
    
    # Get prediction data
    y_test = model_data.get("y_test")
    y_pred = model_data.get("y_pred")
    model = model_data.get("model")
    
    # Calculate prediction probabilities for ROC curve
    y_prob = None
    try:
        from sklearn.metrics import roc_curve, auc
        if hasattr(model, "predict_proba"):
            scaler = model_data.get("scaler")
            X_test = model_data.get("X_test")
            if scaler is not None and X_test is not None:
                y_prob = model.predict_proba(scaler.transform(X_test))[:,1]
            else:
                y_prob = y_pred
        else:
            y_prob = y_pred
    except Exception as e:
        st.warning(f"Could not calculate ROC curve: {e}")
    
    # Create tabs for different evaluation aspects
    tab_names = ["Performance Metrics", "Feature Importance"]
    if y_prob is not None:
        tab_names.append("ROC Curve")
        
    tabs = st.tabs(tab_names)
    
    # Tab 1: Performance Metrics
    with tabs[0]:
        st.subheader("Model Performance Metrics")
        
        # Display confusion matrix and metrics side by side
        cm_col1, cm_col2 = st.columns(2)
        
        with cm_col1:
            st.markdown("**Confusion Matrix**")
            
            # Convert confusion matrix to DataFrame format for plotly
            conf_matrix = model_data.get("confusion_matrix")
            if conf_matrix is not None:
                conf_matrix_values = conf_matrix.values
                
                # Create a plotly heatmap for confusion matrix
                categories = ['Legitimate', 'Fraud']
                
                # Create annotation text
                annotations = []
                for i, row in enumerate(conf_matrix_values):
                    for j, value in enumerate(row):
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=str(value),
                                font=dict(color='white' if value > conf_matrix_values.max()/2 else 'black'),
                                showarrow=False
                            )
                        )
                
                fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix_values,
                    x=['Predicted<br>Legitimate', 'Predicted<br>Fraud'],
                    y=['Actual<br>Legitimate', 'Actual<br>Fraud'],
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig.update_layout(
                    title='Confusion Matrix',
                    annotations=annotations,
                    height=400,
                    width=500,
                    xaxis=dict(title='Predicted'),
                    yaxis=dict(title='Actual')
                )
                
                st.plotly_chart(fig)
            else:
                st.error("Confusion matrix not available")
        
        with cm_col2:
            # Calculate and display metrics
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Create metrics display
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Value': [accuracy, precision, recall, f1]
                }
                
                # Create a bar chart for metrics
                fig = px.bar(
                    metrics_data,
                    x='Metric',
                    y='Value',
                    color='Metric',
                    title='Model Performance Metrics',
                    text_auto='.2f'
                )
                
                fig.update_layout(
                    yaxis_range=[0, 1],
                    height=400
                )
                
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
        
        # Add metric explanations
        with st.expander("Metric Explanations"):
            st.markdown("""
            - **Accuracy**: Proportion of correct predictions (both true positives and true negatives)
            - **Precision**: Proportion of positive predictions that are actually positive (TP / (TP + FP))
            - **Recall**: Proportion of actual positives that are correctly identified (TP / (TP + FN))
            - **F1 Score**: Harmonic mean of precision and recall, balancing both metrics
            
            For fraud detection, high precision means fewer false alarms, while high recall means catching more fraudulent claims.
            """)
    
    # Tab 2: Feature Importance
    with tabs[1]:
        st.subheader("Feature Importance Analysis")
        
        try:
            # Get feature importance
            feature_columns = model_data.get("feature_columns", [])
            feature_columns_exist = feature_columns is not None and hasattr(feature_columns, '__len__') and len(feature_columns) > 0
            
            if not feature_columns_exist:
                st.error("Feature columns not available")
                return
                
            # Ensure feature_columns is a list
            if hasattr(feature_columns, 'tolist'):
                feature_columns = feature_columns.tolist()
                
            coefs = model.coef_[0]
            
            # Make sure we have same number of features and coefficients
            if len(feature_columns) == len(coefs):
                feature_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Coefficient': coefs,
                    'Absolute Importance': np.abs(coefs)
                }).sort_values('Absolute Importance', ascending=False)
                
                # Add feature descriptions
                feature_desc = {
                    'ind_count': 'Number of individuals involved',
                    'amount_paid': 'Total amount paid ($)',
                    'num_frauds_community': 'Frauds in same community',
                    'num_frauds_neighborhood': 'Frauds in neighborhood (within 4 hops)',
                    'influence': 'PageRank centrality score',
                    'policy_premium_A': 'Premium type A',
                    'policy_premium_B': 'Premium type B',
                    'policy_premium_C': 'Premium type C',
                    'policy_premium_D': 'Premium type D',
                    'policy_premium_U': 'Premium type unknown'
                }
                
                feature_importance['Description'] = feature_importance['Feature'].map(lambda x: feature_desc.get(x, x))
                
                # Plot feature importance with Plotly
                colors = ['red' if x > 0 else 'green' for x in feature_importance['Coefficient']]
                
                fig = px.bar(
                    feature_importance,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Coefficients)',
                    hover_data=['Description', 'Absolute Importance'],
                    labels={'Coefficient': 'Impact on Fraud Probability'},
                    color='Coefficient', 
                    color_continuous_scale=['green', 'white', 'red'],
                    color_continuous_midpoint=0
                )
                
                # Add a vertical line at x=0
                fig.add_shape(
                    type="line",
                    x0=0, y0=-0.5,
                    x1=0, y1=len(feature_importance)-0.5,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                fig.update_layout(
                    height=600,
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig)
                
                # Feature importance explanation
                st.markdown("""
                **How to interpret this chart:**
                - **Red bars (positive coefficients)**: These features increase the probability of fraud
                - **Green bars (negative coefficients)**: These features decrease the probability of fraud
                - **Bar length**: Indicates the strength of the effect
                
                The model uses logistic regression, so each coefficient represents the change in log-odds of fraud for a one-unit increase in the feature.
                """)
                
                # Feature importance table with AgGrid
                st.subheader("Feature Importance Table")
                
                # Prepare display columns
                display_df = feature_importance[['Feature', 'Description', 'Coefficient', 'Absolute Importance']].reset_index(drop=True)
                
                # Configure the grid with cell highlighting based on coefficient values
                gb = GridOptionsBuilder.from_dataframe(display_df)
                gb.configure_column("Coefficient", cellStyle=JsCode("""
                    function(params) {
                        if (params.value > 0) {
                            return {'color': 'red'};
                        } else if (params.value < 0) {
                            return {'color': 'green'};
                        }
                        return null;
                    }
                """))
                gb.configure_column("Absolute Importance", type=["numericColumn", "numberColumnFilter"], precision=4)
                gb.configure_column("Coefficient", type=["numericColumn", "numberColumnFilter"], precision=4)
                gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
                
                grid_options = gb.build()
                
                # Display the table with AgGrid
                AgGrid(
                    display_df,
                    gridOptions=grid_options,
                    allow_unsafe_jscode=True,
                    fit_columns_on_grid_load=True,
                    height=400,
                    pagination=True,
                    paginationPageSize=20
                )
            else:
                st.warning(f"Feature columns ({len(feature_columns)}) and coefficients ({len(coefs)}) length mismatch")
        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")
            st.exception(e)
    
    # Tab 3: ROC Curve (if available)
    if y_prob is not None and len(tabs) > 2:
        with tabs[2]:
            st.subheader("ROC Curve Analysis")
            
            try:
                # Calculate ROC curve points
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC curve with Plotly
                fig = go.Figure()
                
                # Add ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC curve (area = {roc_auc:.2f})',
                    line=dict(color='blue', width=2)
                ))
                
                # Add diagonal line (random classifier)
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'),
                    legend=dict(x=0.7, y=0.05),
                    width=700,
                    height=500
                )
                
                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig)
                
                # ROC curve explanation
                st.markdown("""
                **How to interpret the ROC Curve:**
                
                The ROC (Receiver Operating Characteristic) curve shows the trade-off between the true positive rate (TPR, or sensitivity) and false positive rate (FPR) at various threshold settings.
                
                - **Area Under the Curve (AUC)**: A measure of model performance. Higher is better, with 1.0 being perfect and 0.5 being no better than random.
                - **Diagonal line**: Represents random guessing (AUC = 0.5)
                - **Upper left corner**: The ideal point (high TPR, low FPR)
                
                For fraud detection, you can adjust the classification threshold to balance between:
                - Catching more fraud (higher TPR, but more false alarms)
                - Reducing false alarms (lower FPR, but missing more fraud)
                """)
                
                # Create threshold selection visualization
                st.subheader("Threshold Selection Tool")
                
                # Create dataframe with thresholds
                threshold_df = pd.DataFrame({
                    'Threshold': thresholds,
                    'True Positive Rate': tpr,
                    'False Positive Rate': fpr
                })
                if len(thresholds) < len(tpr):
                    # Option 1: Remove the last element from tpr and fpr
                    threshold_df = pd.DataFrame({
                        'Threshold': thresholds,
                        'True Positive Rate': tpr[:-1],  # Exclude the last point
                        'False Positive Rate': fpr[:-1]  # Exclude the last point
                    })
                else:
                    # Option 2: Keep all elements if they're already the same length
                    threshold_df = pd.DataFrame({
                        'Threshold': thresholds,
                        'True Positive Rate': tpr,
                        'False Positive Rate': fpr
                    })
                # Only keep some threshold points for clarity
                if len(threshold_df) > 20:
                    indices = np.linspace(0, len(threshold_df)-1, 20, dtype=int)
                    threshold_df = threshold_df.iloc[indices]
                
                # Create interactive threshold selector
                selected_threshold = st.slider(
                    "Select classification threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )
                
                # Find closest threshold
                closest_idx = (np.abs(threshold_df['Threshold'] - selected_threshold)).argmin()
                selected_tpr = threshold_df.iloc[closest_idx]['True Positive Rate']
                selected_fpr = threshold_df.iloc[closest_idx]['False Positive Rate']
                
                # Display metrics at selected threshold
                col1, col2, col3 = st.columns(3)
                col1.metric("Threshold", f"{selected_threshold:.2f}")
                col2.metric("True Positive Rate", f"{selected_tpr:.2f}")
                col3.metric("False Positive Rate", f"{selected_fpr:.2f}")
                
                # Create a visual representation
                fig = go.Figure()
                
                # Add ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name='ROC curve',
                    line=dict(color='blue', width=2)
                ))
                
                # Add selected point
                fig.add_trace(go.Scatter(
                    x=[selected_fpr],
                    y=[selected_tpr],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name=f'Threshold = {selected_threshold:.2f}'
                ))
                
                # Add diagonal line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Selected Threshold on ROC Curve',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'),
                    legend=dict(x=0.7, y=0.05),
                    width=700,
                    height=500
                )
                
                st.plotly_chart(fig)
                
                # Display threshold data in an AgGrid table
                st.subheader("Threshold Data")
                
                threshold_display = threshold_df.round(4)
                
                # Configure the grid
                gb = GridOptionsBuilder.from_dataframe(threshold_display)
                gb.configure_column("Threshold", type=["numericColumn", "numberColumnFilter"], precision=4)
                gb.configure_column("True Positive Rate", type=["numericColumn", "numberColumnFilter"], precision=4)
                gb.configure_column("False Positive Rate", type=["numericColumn", "numberColumnFilter"], precision=4)
                gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
                
                grid_options = gb.build()
                
                # Display with AgGrid
                AgGrid(
                    threshold_display,
                    gridOptions=grid_options,
                    fit_columns_on_grid_load=True,
                    height=400,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=True,
                    pagination=True,
                    paginationPageSize=20
                )
                
            except Exception as e:
                st.error(f"Error in ROC curve analysis: {e}")
                st.exception(e)

# Main app structure
def main():
    """Main app function"""
    st.title("Insurance Fraud Detection System")
    
    db = get_db_connection()
    if not db:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Explore Claims", "Fraud Detection", "Model Evaluation"])
    
    if page == "Overview":
        display_overview(db)
    elif page == "Explore Claims":
        explore_claims(db)
    elif page == "Fraud Detection":
        fraud_detection(db)
    elif page == "Model Evaluation":
        model_evaluation(db)

if __name__ == "__main__":
    main()