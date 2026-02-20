from pbi_client import run_dax

dax_query = """
EVALUATE
TOPN(10, 'Query1')
"""

print("Connecting to Power BI semantic model...")
print("Auth sources: PBI_ACCESS_TOKEN or (PBI_CLIENT_ID + PBI_CLIENT_SECRET [+ PBI_TENANT_ID]).")

df = run_dax(dax_query)

print("\nSUCCESS - Data pulled from Power BI:\n")
print(df.head(10))
