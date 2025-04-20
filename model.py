"""
Model module for fraud detection
Contains functions for training and evaluating the fraud detection model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feature store class (duplicated from utils to avoid circular imports)
class FeatureStore:
    IND_COUNT = "ind_count"
    AMT_PAID = "amount_paid"
    POL_EXPIRED = "policy_expired"
    POL_PREMIUM = "policy_premium"
    INFLUENCE = "influence"
    NUM_FRAUDS_NEIGHBORHOOD = "num_frauds_neighborhood"
    NUM_FRAUDS_COMMUNITY = "num_frauds_community"
    EMBEDDING = "embedding"
    FRAUD = 'fraud'

def get_individual_count(db, claim_id):
    """Count individuals related to a claim"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[]->(incident:INCIDENT)-[]->(individual:INDIVIDUAL)
            RETURN COUNT(individual) AS ind_count
        """))
        return result[0]["ind_count"] if result else 0
    except Exception as e:
        logger.error(f"Error getting individual count for claim {claim_id}: {e}")
        return 0

def get_amount_paid(db, claim_id):
    """Get total amount paid for a claim"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})<-[]-(payment:CLAIM_PAYMENT)
            RETURN SUM(payment.amount) AS amount_paid
        """))
        amt_paid = result[0]["amount_paid"] if result else 0
        return amt_paid if amt_paid else 0
    except Exception as e:
        logger.error(f"Error getting amount paid for claim {claim_id}: {e}")
        return 0

def get_policy_premium(db, claim_id):
    """Get policy premium type for a claim"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[]->(:INCIDENT)-[]->(policy:POLICY)
            RETURN policy.premium AS policy_premium 
        """))
        
        if result and len(result) > 0:
            premium = result[0].get("policy_premium")
            return premium if premium else "U"
        else:
            return "U"
    except Exception as e:
        logger.error(f"Error getting policy premium for claim {claim_id}: {e}")
        return "U"

def get_num_frauds_community(db, claim_id):
    """Count frauds in the same community as the claim"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}}), (fraud:CLAIM {{fraud: True}})
            WHERE claim.community = fraud.community
            RETURN COUNT(fraud) AS num_frauds_community
        """))
        return result[0]["num_frauds_community"] if result else 0
    except Exception as e:
        logger.error(f"Error getting frauds in community for claim {claim_id}: {e}")
        return 0

def get_num_frauds_neighborhood(db, claim_id):
    """Count frauds in the neighborhood (up to 4 hops away)"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[*bfs ..4]-(fraud:CLAIM {{fraud: True}})
            RETURN COUNT(fraud) AS num_frauds_neighborhood
        """))
        return result[0]["num_frauds_neighborhood"] if result else 0
    except Exception as e:
        logger.error(f"Error getting frauds in neighborhood for claim {claim_id}: {e}")
        return 0

def get_influence(db, claim_id):
    """Get influence score (PageRank) for a claim"""
    try:
        result = list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})
            RETURN claim.influence AS influence
        """))
        
        if result and len(result) > 0 and result[0].get("influence") is not None:
            influence = result[0].get("influence")
            return float(influence)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error getting influence for claim {claim_id}: {e}")
        return 0.0

def get_claims(db, limit=1000):
    """Get claims with fraud status"""
    try:
        return list(db.execute_and_fetch(f"""
            MATCH (claim:CLAIM)
            WHERE claim.clm_id IS NOT NULL
            RETURN claim.clm_id AS claim_id, claim.fraud AS fraud
            LIMIT {limit}
        """))
    except Exception as e:
        logger.error(f"Error getting claims: {e}")
        return []

def extract_features(db, claim_id):
    """Extract all features for a claim"""
    try:
        features = {
            FeatureStore.IND_COUNT: get_individual_count(db, claim_id),
            FeatureStore.AMT_PAID: get_amount_paid(db, claim_id),
            FeatureStore.NUM_FRAUDS_COMMUNITY: get_num_frauds_community(db, claim_id),
            FeatureStore.NUM_FRAUDS_NEIGHBORHOOD: get_num_frauds_neighborhood(db, claim_id),
            FeatureStore.INFLUENCE: get_influence(db, claim_id),
            FeatureStore.POL_PREMIUM: get_policy_premium(db, claim_id),
        }
        return features
    except Exception as e:
        logger.error(f"Error extracting features for claim {claim_id}: {e}")
        return {}

def train_fraud_detection_model(db):
    """
    Train a fraud detection model using data from the graph database
    
    Parameters:
    db - Memgraph database connection
    
    Returns:
    Dictionary containing model, scaler, feature columns, and evaluation metrics
    """
    logger.info("Starting fraud detection model training")
    
    # Get all claims
    claims_data = get_claims(db)
    logger.info(f"Retrieved {len(claims_data)} claims from database")
    
    if not claims_data:
        raise ValueError("No claims found in database")
    
    # Create feature store
    feature_rows = []
    feature_columns = [
        FeatureStore.IND_COUNT, 
        FeatureStore.AMT_PAID,
        FeatureStore.NUM_FRAUDS_COMMUNITY,
        FeatureStore.NUM_FRAUDS_NEIGHBORHOOD,
        FeatureStore.INFLUENCE,
        FeatureStore.POL_PREMIUM,
        FeatureStore.FRAUD,
        "claim_id"
    ]
    
    # Extract features for each claim
    for result in claims_data:
        claim_id = result.get("claim_id")
        fraud = result.get("fraud")
        
        # Skip claims without ID or fraud status
        if claim_id is None or fraud is None:
            continue
        
        try:
            # Use extract_features function to get all features
            features = extract_features(db, claim_id)
            
            # Add fraud status and claim_id
            features[FeatureStore.FRAUD] = fraud
            features["claim_id"] = claim_id
            
            # Add to feature store
            feature_rows.append(features)
        except Exception as e:
            logger.warning(f"Error extracting features for claim {claim_id}: {e}")
    
    # Convert to DataFrame
    if not feature_rows:
        raise ValueError("Failed to extract features for any claims")
    
    feature_store = pd.DataFrame(feature_rows)
    logger.info(f"Created feature store with {len(feature_store)} rows")
    
    # Drop any rows with missing values
    feature_store = feature_store.dropna()
    logger.info(f"After dropping NA values: {len(feature_store)} rows")
    
    # Set claim_id as index
    feature_store = feature_store.set_index("claim_id")
    
    # Prepare features for training
    X = feature_store[[
        FeatureStore.IND_COUNT, 
        FeatureStore.AMT_PAID,
        FeatureStore.NUM_FRAUDS_COMMUNITY,
        FeatureStore.NUM_FRAUDS_NEIGHBORHOOD,
        FeatureStore.INFLUENCE,
        FeatureStore.POL_PREMIUM,
    ]]
    
    # Convert data types
    X = X.astype({
        FeatureStore.IND_COUNT: int,
        FeatureStore.AMT_PAID: float,
        FeatureStore.NUM_FRAUDS_COMMUNITY: int,
        FeatureStore.NUM_FRAUDS_NEIGHBORHOOD: int,
        FeatureStore.INFLUENCE: float,
        FeatureStore.POL_PREMIUM: "category",
    })
    
    # One-hot encode policy premium
    dummies = pd.get_dummies(X[FeatureStore.POL_PREMIUM], prefix=FeatureStore.POL_PREMIUM)
    X = X.join(dummies)
    X = X.drop(columns=[FeatureStore.POL_PREMIUM])
    
    # Get the target variable
    y = feature_store[FeatureStore.FRAUD].astype(bool)
    
    # Verify that X and y have same dimensions
    if len(X) != len(y):
        logger.error(f"X and y have different lengths: X={len(X)}, y={len(y)}")
        # Align the indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        logger.info(f"Aligned X and y to common indices: {len(X)} samples")
    
    # Feature columns to keep for prediction
    feature_columns = X.columns
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Upsample minority class
    if len(y_train[y_train == True]) > 0:
        train_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_data['fraud'] = y_train.values
        
        fraud_cases = train_data[train_data['fraud'] == True]
        non_fraud_cases = train_data[train_data['fraud'] == False]
        
        logger.info(f"Before upsampling: {len(fraud_cases)} fraud cases, {len(non_fraud_cases)} non-fraud cases")
        
        # Balance the dataset if needed
        if len(fraud_cases) < len(non_fraud_cases) // 2:
            fraud_cases_upsampled = resample(
                fraud_cases,
                replace=True,
                n_samples=len(non_fraud_cases) // 2,
                random_state=42
            )
            train_data = pd.concat([non_fraud_cases, fraud_cases_upsampled])
            
            y_train = train_data['fraud']
            X_train_scaled = train_data.drop('fraud', axis=1).values
            
            logger.info(f"After upsampling: {len(y_train[y_train == True])} fraud cases, {len(y_train[y_train == False])} non-fraud cases")
    
    # Train model
    logger.info("Training logistic regression model")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Get model evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual Legitimate', 'Actual Fraud'], 
        columns=['Predicted Legitimate', 'Predicted Fraud']
    )
    
    logger.info("Model training completed successfully")
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "confusion_matrix": cm_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }

def predict_fraud(model_data, claim_features):
    """
    Predict fraud for a claim using the trained model
    
    Parameters:
    model_data - Dictionary with model and scaler
    claim_features - DataFrame with features for the claim
    
    Returns:
    Dictionary with fraud prediction and probability
    """
    try:
        model = model_data["model"]
        scaler = model_data["scaler"]
        
        # Ensure all columns match the training data
        required_columns = model_data["feature_columns"]
        for col in required_columns:
            if col not in claim_features.columns:
                claim_features[col] = 0
        
        # Keep only required columns in the same order as training
        claim_features = claim_features[required_columns]
        
        # Replace any NaN values with 0
        claim_features = claim_features.fillna(0)
        
        # Scale the features
        features_scaled = scaler.transform(claim_features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return {
            "is_fraud": prediction[0],
            "fraud_probability": probability[0][1] if len(probability[0]) > 1 else 0
        }
    except Exception as e:
        logger.error(f"Error making fraud prediction: {e}")
        return {
            "is_fraud": False,
            "fraud_probability": 0,
            "error": str(e)
        }

def prepare_new_claim_features(manual_features):
    """
    Prepare features for a manually entered claim
    
    Parameters:
    manual_features - Dictionary with manually entered feature values
    
    Returns:
    DataFrame with properly formatted features
    """
    try:
        # Start with basic features
        features_dict = {
            FeatureStore.IND_COUNT: manual_features.get("ind_count", 0),
            FeatureStore.AMT_PAID: manual_features.get("amount_paid", 0),
            FeatureStore.NUM_FRAUDS_COMMUNITY: manual_features.get("num_frauds_community", 0),
            FeatureStore.NUM_FRAUDS_NEIGHBORHOOD: manual_features.get("num_frauds_neighborhood", 0),
            FeatureStore.INFLUENCE: manual_features.get("influence", 0),
        }
        
        # Add one-hot encoded premium type
        premium_type = manual_features.get("premium_type", "U")
        for p_type in ["A", "B", "C", "D", "U"]:
            features_dict[f"policy_premium_{p_type}"] = 1 if p_type == premium_type else 0
        
        # Create DataFrame
        features = pd.DataFrame([features_dict])
        
        return features
    except Exception as e:
        logger.error(f"Error preparing manual claim features: {e}")
        # Return empty DataFrame with expected columns
        empty_df = pd.DataFrame(columns=[
            FeatureStore.IND_COUNT,
            FeatureStore.AMT_PAID,
            FeatureStore.NUM_FRAUDS_COMMUNITY,
            FeatureStore.NUM_FRAUDS_NEIGHBORHOOD,
            FeatureStore.INFLUENCE,
            "policy_premium_A",
            "policy_premium_B",
            "policy_premium_C",
            "policy_premium_D",
            "policy_premium_U"
        ])
        return empty_df