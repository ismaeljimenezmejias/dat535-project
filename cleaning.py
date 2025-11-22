import pandas as pd
from google.cloud import storage

# === 1. Google Cloud Storage ===
client = storage.Client()
bucket_name = "medallion-dat535"

# Paths
silver_input_path = "silver/mental_health_structured.csv"
silver_output_path = "silver/mental_health_clean.csv"

bucket = client.bucket(bucket_name)

# === 2. Load structured CSV from GCS ===
blob_input = bucket.blob(silver_input_path)
blob_input.download_to_filename("mental_health_structured.csv")

df = pd.read_csv("mental_health_structured.csv")

print(f"✅ Loaded structured CSV: {silver_input_path}")

# === 3. Null value summary ===
null_counts = df.isnull().sum()
total_rows = len(df)

print("\n=== Null Value Summary ===")
for col, n in null_counts.items():
    if n > 0:
        print(f"{col}: {n} nulls ({n/total_rows:.2%})")

# === 4. Remove all rows with any null value ===
df_clean = df.dropna(how="any")
print(f"\nRows before cleaning: {len(df)}")
print(f"Rows after cleaning: {len(df_clean)}")
print(f"Rows removed: {len(df)-len(df_clean)} ({(len(df)-len(df_clean))/len(df):.2%})")

# === 5. Normalize text case ===
text_cols = df_clean.select_dtypes(include="object").columns
df_clean[text_cols] = df_clean[text_cols].apply(lambda x: x.str.title())

# === 6. Drop redundant SocialWeaknessSecondary column if identical ===
if "SocialWeaknessSecondary" in df_clean.columns:
    if (df_clean["SocialWeaknessPrimary"] == df_clean["SocialWeaknessSecondary"]).all():
        df_clean = df_clean.drop(columns=["SocialWeaknessSecondary"])
        df_clean = df_clean.rename(columns={"SocialWeaknessPrimary": "SocialWeakness"})
        print("✅ SocialWeaknessSecondary dropped and SocialWeaknessPrimary renamed.")

# === 7. Save cleaned CSV locally ===
df_clean.to_csv("mental_health_clean.csv", index=False, encoding="utf-8")
print(f"💾 Cleaned CSV saved locally: mental_health_clean.csv")

# === 8. Upload cleaned CSV to GCS (Gold layer) ===
gold_output_path = "gold/mental_health_clean.csv"
blob_output = bucket.blob(gold_output_path)
blob_output.upload_from_filename("mental_health_clean.csv")

print(f"🏆 Uploaded cleaned CSV to GOLD layer: {gold_output_path}")

# === 9. Preview ===
print("\nPreview:")
print(df_clean.head(5).to_markdown())
