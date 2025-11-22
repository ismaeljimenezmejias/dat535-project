import json
import pandas as pd
from google.cloud import storage

# === 1. Google Cloud Storage ===
client = storage.Client()

bucket_name = "medallion-dat535"
blob_path = "bronce/raw/mental_health_unstructured.jsonl"

bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)

print("📥 Downloading JSONL from GCS...")
content = blob.download_as_text(encoding="utf-8").split("\n")

# === 2. Read JSONL ===
records = []
for line in content:
    if not line.strip():
        continue
    try:
        records.append(json.loads(line.strip()))
    except json.JSONDecodeError:
        print("❌ Error reading a JSON line, skipping...")

print(f"✅ Loaded {len(records)} records.")

# === 3. Flatten structures ===
flat_rows = []
for r in records:
    person = r.get("person", {})
    status = r.get("status", {})
    lifestyle = r.get("lifestyle", {})

    flat = {
        "Gender": person.get("gender"),
        "Country": person.get("country"),
        "Occupation": person.get("occupation"),
        "SelfEmployed": status.get("self_employed"),
        "FamilyHistory": status.get("family_history"),
        "Treatment": status.get("treatment"),
        "MentalHealthHistory": status.get("mental_health_history"),
        "IncreasingStress": status.get("increasing_stress"),
        "MoodSwings": status.get("mood_swings"),
        "SocialWeaknessPrimary": status.get("social_weakness_primary"),
        "SocialWeaknessSecondary": status.get("social_weakness_secondary"),
        "CopingStruggles": status.get("coping_struggles"),
        "MentalHealthInterview": status.get("mental_health_interview"),
        "CareOptions": status.get("care_options"),
        "DaysIndoors": lifestyle.get("days_indoors"),
        "HabitsChange": lifestyle.get("habits_change"),
        "WorkInterest": lifestyle.get("work_interest"),
    }

    flat_rows.append(flat)

# === 4. Create structured DataFrame ===
df = pd.DataFrame(flat_rows)
print(f"✅ DataFrame created with {df.shape[0]} rows and {df.shape[1]} columns.")

# === 5. Save CSV locally ===
output_csv = "mental_health_structured.csv"
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"💾 CSV saved locally as: {output_csv}")

# === 6. Upload structured CSV to GCS (Silver layer) ===
silver_path = "silver/mental_health_structured.csv"
silver_blob = bucket.blob(silver_path)

silver_blob.upload_from_filename(output_csv)

print(f"🚀 Uploaded structured CSV to GCS: {silver_path}")
