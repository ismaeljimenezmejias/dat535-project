import json
import pandas as pd
from pathlib import Path

# === 1. Paths ===
input_path = Path("/Users/trymfalkum/Desktop/DAT535/mental_health_unstructured_MESSY.jsonl")
output_path = Path("mental_health_structured.csv")

# === 2. Read JSONL ===
records = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            records.append(data)
        except json.JSONDecodeError:
            print("❌ Error reading a JSON line, skipping...")

print(f"✅ Loaded {len(records)} records.")

# === 3. Flatten structures ===
# Each record contains nested sections: person / status / lifestyle
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



# === 5. Save as CSV ===
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"💾 Structured dataset saved to: {output_path.resolve()}")
