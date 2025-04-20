#!/usr/bin/env python3
"""
Script to load the insurance fraud detection demo dataset into Memgraph
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from gqlalchemy import Memgraph
from gqlalchemy.transformations.importing.loaders import ParquetLocalFileSystemImporter

def main():
    """Main function to load the dataset"""
    print("Loading insurance fraud detection demo dataset...")
    
    # Load configuration from environment file
    load_dotenv(".env")

    # Define Memgraph connection
    mg_host = os.getenv("MG_HOST", "localhost")
    mg_port = int(os.getenv("MG_PORT", "7687"))
    
    try:
        print(f"Connecting to Memgraph at {mg_host}:{mg_port}...")
        db = Memgraph(mg_host, mg_port)
        print("Connected successfully.")
    except Exception as e:
        print(f"Error connecting to Memgraph: {e}")
        print("Make sure Memgraph is running and accessible.")
        return
    
    # Load configuration from YAML file
    PATH_TO_CONFIG_YAML = "./config.yml"
    
    try:
        with Path(PATH_TO_CONFIG_YAML).open("r") as f_:
            data_configuration = yaml.safe_load(f_)
    except Exception as e:
        print(f"Error loading configuration from {PATH_TO_CONFIG_YAML}: {e}")
        return
    
    try:
        # Initialize the importer
        print("Initializing data importer...")
        translator = ParquetLocalFileSystemImporter(
            path="./dataset/data/",
            data_configuration=data_configuration
        )
        
        # Load the data
        print("Loading data into Memgraph (this may take a few minutes)...")
        translator.translate(drop_database_on_start=True)
        
        print("Data loaded successfully!")
        
        # Run PageRank and community detection
        print("Running graph analytics algorithms...")
        
        try:
            # Add PageRank scores for influence
            print("Computing PageRank scores...")
            db.execute("""
                CALL pagerank.get() YIELD node, rank
                SET node.influence = rank;
            """)
        except Exception as e:
            print(f"Warning: Couldn't run PageRank algorithm: {e}")
            print("Make sure the MAGE library is installed.")
        
        try:
            # Add community labels
            print("Running community detection...")
            db.execute("""
                CALL community_detection.get() YIELD node, community_id
                SET node.community = community_id;
            """)
        except Exception as e:
            print(f"Warning: Couldn't run community detection: {e}")
            print("Make sure the MAGE library is installed.")
        
        print("Dataset is now ready to use.")
        print("You can now run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

if __name__ == "__main__":
    main()