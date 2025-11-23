from pyspark import SparkContext
import csv
import io

# === 0. Spark ===
sc = SparkContext(appName="MentalHealthCleaningRDD")

# === 1. Read structured CSV from SILVER layer (GCS) ===
silver_path = "gs://medallion-dat535/silver/mental_health_structured.csv"

rdd = sc.textFile(silver_path)
header = rdd.first()
columns = header.split(",")

rdd_rows = rdd.filter(lambda x: x != header)

# === 2. MAP: parse CSV to dict ===
def parse_line(line):
    reader = csv.reader(io.StringIO(line))
    row = next(reader)
    return dict(zip(columns, row))

rdd_dict = rdd_rows.map(parse_line)

# === 3. FILTER: remove rows with nulls ===
def no_nulls(record):
    return all(v not in ("", None, "NULL") for v in record.values())

rdd_nonull = rdd_dict.filter(no_nulls)

# === 4. MAP: Normalize text case ===
def norm(record):
    return {k: (v.title() if isinstance(v, str) else v) for k, v in record.items()}

rdd_norm = rdd_nonull.map(norm)

# === 5. MAP: Merge SocialWeakness columns ===
def merge_sw(r):
    if ("SocialWeaknessPrimary" in r and
        "SocialWeaknessSecondary" in r and
        r["SocialWeaknessPrimary"] == r["SocialWeaknessSecondary"]):
        
        r["SocialWeakness"] = r["SocialWeaknessPrimary"]
        del r["SocialWeaknessPrimary"]
        del r["SocialWeaknessSecondary"]
    return r

rdd_merged = rdd_norm.map(merge_sw)

# === 6. REDUCE: count rows ===
total_rows = rdd_merged.map(lambda _: 1).reduce(lambda a, b: a + b)
print(f"Total cleaned rows: {total_rows}")

# === 7. Convert dict -> CSV line ===
def to_csv(record):
    cols = [c for c in columns if c in record]
    if "SocialWeakness" in record and "SocialWeakness" not in cols:
        cols.append("SocialWeakness")
    return ",".join(record[c] for c in cols)

rdd_csv = rdd_merged.map(to_csv)

# Add updated header
final_columns = list(rdd_merged.first().keys())
header_clean = ",".join(final_columns)
rdd_csv = sc.parallelize([header_clean]).union(rdd_csv)

# === 8. Save final CLEANED CSV to GOLD layer (GCS) ===
gold_path = "gs://medallion-dat535/gold"


rdd_csv.coalesce(1).saveAsTextFile(gold_path + "/csv")

print("Saved csv cleaned dataset to GOLD layer:", gold_path + "/csv")

rdd_df = rdd_merged.toDF()
rdd_df.write.mode("overwrite").parquet(gold_path + "/parquet")

print("Saved parquet cleaned dataset to GOLD layer:", gold_path + "/parquet")

