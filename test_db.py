import pandas as pd
import numpy as np
from src.supabase_client import get_client

df = pd.read_parquet("data/orderbook.parquet").head(2000)

# check for NaN / inf that would break JSON
print("NaN counts per column:")
print(df.isna().sum())
print("any inf:", np.isinf(df.select_dtypes('number')).any().any())

df = df.copy()
df["timestamp"] = df["timestamp"].astype(str)
records = df[["timestamp","bid_price","bid_qty","ask_price","ask_qty"]].to_dict(orient="records")
print("first record:", records[0])
print("num records:", len(records))

sb = get_client()
try:
    r = sb.table("orderbook").insert(records).execute()
    print("BATCH INSERT OK, rows:", len(r.data))
except Exception as e:
    print("BATCH INSERT FAILED:", repr(e))
    # try just the first 5 to narrow it down
    try:
        r = sb.table("orderbook").insert(records[:5]).execute()
        print("first-5 OK:", len(r.data))
    except Exception as e2:
        print("first-5 FAILED:", repr(e2))