import pandas as pd
import numpy as np
from datetime import datetime

class FeatureStore:
    """Class containing feature names used in the fraud detection system"""
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
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[]->(incident:INCIDENT)-[]->(individual:INDIVIDUAL)
        RETURN COUNT(individual) AS {FeatureStore.IND_COUNT}
    """))
    return result[0][FeatureStore.IND_COUNT] if result else 0

def get_amount_paid(db, claim_id):
    """Get total amount paid for a claim"""
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})<-[]-(payment:CLAIM_PAYMENT)
        RETURN SUM(payment.amount) AS {FeatureStore.AMT_PAID}
    """))
    amt_paid = result[0][FeatureStore.AMT_PAID] if result else 0
    return amt_paid if amt_paid else 0

def get_policy_expired(db, claim_id):
    """Check if policy has expired for a claim"""
    today = str(datetime.now().date())
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[]->(:INCIDENT)-[]->(policy:POLICY)
        RETURN policy.end_date < date('{today}') AS {FeatureStore.POL_EXPIRED}
    """))
    return result[0][FeatureStore.POL_EXPIRED] if result else False

def get_policy_premium(db, claim_id):
    """Get policy premium type for a claim"""
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[]->(:INCIDENT)-[]->(policy:POLICY)
        RETURN policy.premium AS {FeatureStore.POL_PREMIUM} 
    """))
    
    if result and len(result) > 0:
        premium = result[0].get(FeatureStore.POL_PREMIUM)
        return premium if premium else "U"
    else:
        return "U"

def get_num_frauds_community(db, claim_id):
    """Count frauds in the same community as the claim"""
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}}), (fraud:CLAIM {{fraud: True}})
        WHERE claim.community = fraud.community
        RETURN COUNT(fraud) AS {FeatureStore.NUM_FRAUDS_COMMUNITY}
    """))
    return result[0][FeatureStore.NUM_FRAUDS_COMMUNITY] if result else 0

def get_num_frauds_neighborhood(db, claim_id):
    """Count frauds in the neighborhood (up to 4 hops away)"""
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})-[*bfs ..4]-(fraud:CLAIM {{fraud: True}})
        RETURN COUNT(fraud) AS {FeatureStore.NUM_FRAUDS_NEIGHBORHOOD}
    """))
    return result[0][FeatureStore.NUM_FRAUDS_NEIGHBORHOOD] if result else 0

def get_influence(db, claim_id):
    """Get influence score (PageRank) for a claim"""
    result = list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM {{clm_id: '{claim_id}'}})
        RETURN claim.influence AS {FeatureStore.INFLUENCE}
    """))
    
    if result and len(result) > 0:
        influence = result[0].get(FeatureStore.INFLUENCE)
        return float(influence) if influence is not None else 0.0
    else:
        return 0.0

def extract_features(db, claim_id, columns):
    """Extract all features for a claim"""
    features = {
        FeatureStore.IND_COUNT: get_individual_count(db, claim_id),
        FeatureStore.AMT_PAID: get_amount_paid(db, claim_id),
        FeatureStore.NUM_FRAUDS_COMMUNITY: get_num_frauds_community(db, claim_id),
        FeatureStore.NUM_FRAUDS_NEIGHBORHOOD: get_num_frauds_neighborhood(db, claim_id),
        FeatureStore.INFLUENCE: get_influence(db, claim_id),
        FeatureStore.POL_PREMIUM: get_policy_premium(db, claim_id),
    }
    
    X = pd.DataFrame([features], index=[claim_id])
    
    X = X.astype({
        FeatureStore.IND_COUNT: int,
        FeatureStore.AMT_PAID: float,
        FeatureStore.NUM_FRAUDS_COMMUNITY: int,
        FeatureStore.NUM_FRAUDS_NEIGHBORHOOD: int,
        FeatureStore.INFLUENCE: float,
        FeatureStore.POL_PREMIUM: "category",
    })
    
    dummies = pd.get_dummies(X[FeatureStore.POL_PREMIUM], prefix=FeatureStore.POL_PREMIUM)
    X = X.join(dummies)
    X = X.drop(columns=[FeatureStore.POL_PREMIUM])
    
    X = X.reindex(columns=columns, fill_value=0)
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X

def get_claims(db, limit=100):
    """Get claims with fraud status"""
    return list(db.execute_and_fetch(f"""
        MATCH (claim:CLAIM)
        RETURN claim.clm_id AS claim_id, claim.fraud AS fraud
        LIMIT {limit}
    """))

def get_claim_details(db, claim_id):
    """Get detailed information about a claim"""
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
        "claim_info": list(claim_info),
        "individuals": list(individuals),
        "payments": list(payments)
    }

def clean_features_for_prediction(features_df):
    """Ensure all features are clean for prediction"""
    clean_df = features_df.copy()
    clean_df = clean_df.fillna(0)
    clean_df = clean_df.replace([np.inf, -np.inf], 0)
    
    for col in clean_df.columns:
        try:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)
        except:
            # Skip columns that can't be converted to numeric
            pass
    
    return clean_df