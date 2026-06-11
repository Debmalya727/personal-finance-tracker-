import os
import sys
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"d:\Projects\Done\personal-finance-tracker\.env")

# Databases to update
db_uris = {}

# 1. Local Database
local_uri = os.getenv("DATABASE_URI")
if local_uri:
    db_uris["Local MySQL"] = local_uri

# 2. Clever Cloud Database
clever_uri = os.getenv("MYSQL_ADDON_URI")
if clever_uri:
    # Use pymysql driver
    if not clever_uri.startswith("mysql+pymysql://"):
        clever_uri = clever_uri.replace("mysql://", "mysql+pymysql://")
    db_uris["Clever Cloud (Production)"] = clever_uri

print("Found databases to update:", list(db_uris.keys()))

for name, uri in db_uris.items():
    print(f"\n--- Updating schema for {name} ---")
    try:
        engine = create_engine(uri)
        inspector = inspect(engine)
        
        # Check BusinessClient columns
        client_cols = [c['name'] for c in inspector.get_columns("business_client")]
        
        # Check BusinessTransaction columns
        tx_cols = [c['name'] for c in inspector.get_columns("business_transaction")]
        
        with engine.begin() as conn:
            # 1. business_client missing columns
            new_client_fields = {
                "name": "VARCHAR(100) NULL",
                "email": "VARCHAR(100) NULL",
                "phone": "VARCHAR(50) NULL",
                "company": "VARCHAR(100) NULL"
            }
            for col_name, col_type in new_client_fields.items():
                if col_name not in client_cols:
                    print(f"Adding '{col_name}' to 'business_client'...")
                    conn.execute(text(f"ALTER TABLE business_client ADD COLUMN {col_name} {col_type};"))
                else:
                    print(f"Column '{col_name}' already exists in 'business_client'.")
            
            # 2. business_transaction missing columns
            new_tx_fields = {
                "client_id": "INT NULL",
                "invoice_status": "VARCHAR(20) NULL DEFAULT 'paid'",
                "due_date": "DATE NULL"
            }
            for col_name, col_type in new_tx_fields.items():
                if col_name not in tx_cols:
                    print(f"Adding '{col_name}' to 'business_transaction'...")
                    conn.execute(text(f"ALTER TABLE business_transaction ADD COLUMN {col_name} {col_type};"))
                else:
                    print(f"Column '{col_name}' already exists in 'business_transaction'.")
            
            # 3. foreign key constraint on business_transaction.client_id
            # First, check if the constraint already exists
            fks = inspector.get_foreign_keys("business_transaction")
            fk_exists = any(fk['constrained_columns'] == ['client_id'] for fk in fks)
            if not fk_exists:
                print("Adding foreign key constraint for 'client_id' on 'business_transaction'...")
                try:
                    conn.execute(text("ALTER TABLE business_transaction ADD CONSTRAINT fk_business_transaction_client FOREIGN KEY (client_id) REFERENCES business_client(id) ON DELETE SET NULL;"))
                except Exception as ex:
                    print(f"Warning: Could not add foreign key constraint (it might already exist under a different name): {ex}")
            else:
                print("Foreign key constraint for 'client_id' already exists.")
                
        print(f"Successfully completed update for {name}!")
    except Exception as e:
        print(f"Error updating schema for {name}: {e}")
