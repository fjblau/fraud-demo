from gqlalchemy import Node, Field
from typing import Optional, List
from datetime import date

class Individual(Node):
    """Model for INDIVIDUAL nodes in the graph database."""
    __primarylabel__ = "INDIVIDUAL"
    __primarykey__ = "ind_id"
    labels = {"INDIVIDUAL"}  # Add this line
    
    ind_id: str = Field()
    first_name: str = Field()
    last_name: str = Field()
    email: Optional[str] = Field()
    phone: Optional[str] = Field()
    date_of_birth: Optional[date] = Field()
    # Add any other fields that exist in your data

class Address(Node):
    """Model for ADDRESS nodes in the graph database."""
    __primarylabel__ = "ADDRESS"
    __primarykey__ = "add_id"
    labels = {"ADDRESS"}  # Add this line

    add_id: str = Field()
    street: str = Field()
    city: str = Field()
    state: str = Field()
    zip_code: str = Field()
    # Add any other fields that exist in your data

class Policy(Node):
    """Model for POLICY nodes in the graph database."""
    __primarylabel__ = "POLICY"
    __primarykey__ = "pol_id"
    labels = {"POLICY"}  # Add this line
    
    pol_id: str = Field()
    start_date: date = Field()
    end_date: date = Field()
    type: str = Field()
    premium: float = Field()
    insurer_id: str = Field()
    insured_with_id: str = Field()
    veh_id: Optional[str] = Field()
    add_id: Optional[str] = Field()
    # Add any other fields that exist in your data

class Vehicle(Node):
    """Model for VEHICLE nodes in the graph database."""
    __primarylabel__ = "VEHICLE"
    __primarykey__ = "veh_id"
    labels = {"VEHICLE"}  # Add this line
    
    veh_id: str = Field()
    make: str = Field()
    model: str = Field()
    year: int = Field()
    vin: str = Field()
    # Add any other fields that exist in your data

class Incident(Node):
    """Model for INCIDENT nodes in the graph database."""
    __primarylabel__ = "INCIDENT"
    __primarykey__ = "inc_id"
    labels = {"INCIDENT"}  # Add this line
    
    inc_id: str = Field()
    date: date = Field()
    description: str = Field()
    pol_id: str = Field()
    add_id: str = Field()
    # Add any other fields that exist in your data

class Claim(Node):
    """Model for CLAIM nodes in the graph database."""
    __primarylabel__ = "CLAIM"
    __primarykey__ = "clm_id"
    labels = {"CLAIM"}  # Add this line
    
    clm_id: str = Field()
    amount: float = Field()
    status: str = Field()
    inc_id: str = Field()
    # Add any other fields that exist in your data

class ClaimPayment(Node):
    """Model for CLAIM_PAYMENT nodes in the graph database."""
    __primarylabel__ = "CLAIM_PAYMENT"
    __primarykey__ = "pay_id"
    labels = {"CLAIM_PAYMENT"}  # Add this line
    
    pay_id: str = Field()
    amount: float = Field()
    date: date = Field()
    payer_id: str = Field()
    payee_id: str = Field()
    clm_id: str = Field()
    # Add any other fields that exist in your data

class Injury(Node):
    """Model for INJURY nodes in the graph database."""
    __primarylabel__ = "INJURY"
    __primarykey__ = "inj_id"
    labels = {"INJURY"}  # Add this line
    
    inj_id: str = Field()
    description: str = Field()
    severity: str = Field()
    clm_id: str = Field()
    ind_id: str = Field()
    # Add any other fields that exist in your data