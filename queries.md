# Cypher Queries from Insurance Fraud Detection App

These queries are used in the application to interact with the Memgraph graph database.

## Database Statistics Queries

### Count Total Claims
```cypher
MATCH (c:CLAIM) 
RETURN count(c) as count
```

### Count Fraudulent Claims
```cypher
MATCH (c:CLAIM {fraud: true}) 
RETURN count(c) as count
```

### Count Individuals
```cypher
MATCH (i:INDIVIDUAL) 
RETURN count(i) as count
```

### Count Policies
```cypher
MATCH (p:POLICY) 
RETURN count(p) as count
```

## Claims Exploration Queries

### Get Claims with Individuals and Incident Dates
```cypher
MATCH (c:CLAIM)
OPTIONAL MATCH (c)-[:ON_INCIDENT]->(incident:INCIDENT)
OPTIONAL MATCH (incident)-[:INCIDENT]->(individual:INDIVIDUAL)
WITH c, 
    COLLECT(DISTINCT individual.last_name) AS last_names,
    COLLECT(DISTINCT toString(incident.accident_date)) AS incident_dates
RETURN c.clm_id AS claim_id, 
    c.amount AS amount, 
    c.fraud AS is_fraud,
    last_names AS individuals_last_names,
    incident_dates AS dates
LIMIT 100
```

### Get Claim Details
```cypher
MATCH (claim:CLAIM {clm_id: '{claim_id}'})-[:ON_INCIDENT]->(incident:INCIDENT)
RETURN claim, incident
```

### Get Individuals Involved in a Claim
```cypher
MATCH (claim:CLAIM {clm_id: '{claim_id}'})-[:ON_INCIDENT]->(incident:INCIDENT)-[:INCIDENT]->(individual:INDIVIDUAL)
RETURN individual.first_name AS first_name, individual.last_name AS last_name
```

### Get Payments for a Claim
```cypher
MATCH (payment:CLAIM_PAYMENT)-[:ON_CLAIM]->(claim:CLAIM {clm_id: '{claim_id}'})
RETURN payment.amount AS amount, payment.pay_id AS payment_id
```

## Graph Visualization Query

### Visualize Claim with Connected Entities (2 hops)
```cypher
MATCH p=(claim:CLAIM {clm_id: '{claim_id}'})-[*1..2]-()
RETURN p
```

Note: In the actual application, the `{claim_id}` placeholders are replaced with the actual claim ID selected by the user.