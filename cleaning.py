from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MentalHealthCleaningRDD") \
    .getOrCreate()

sc = spark.sparkContext

# === 1. Read structured CSV from SILVER layer (GCS) ===
silver_path = "gs://medallion-dat535/silver/mental_health_structured.csv"

rdd = sc.textFile(silver_path)
header = rdd.first()
columns = header.split(",")

rdd_rows = rdd.filter(lambda x: x != header)

# === 2. MAP: parse CSV to dict ===
def parse_line(line):
    
    import csv
    import io
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

# Convierte RDD a DataFrame
rdd_df = rdd_merged.toDF()

# Convierte RDD a DataFrame
rdd_df = rdd_merged.toDF()

# Reparticiona a 10 particiones
rdd_df_small = rdd_df.repartition(50)

# Guarda Parquet
rdd_df_small.write.mode("overwrite").parquet(gold_path + "/mental_health_clean.parquet")
print("Saved parquet cleaned dataset to GOLD layer:", gold_path + "/mental_health_clean.parquet")

# Guarda CSV
rdd_df_small.write.mode("overwrite").option("header", True).csv(gold_path + "/mental_health_clean.csv")
print("Saved csv cleaned dataset to GOLD layer:", gold_path + "/mental_health_clean.csv")

# Guarda JSON
rdd_df_small.write.mode("overwrite").json(gold_path + "/mental_health_clean.json")
print("Saved json cleaned dataset to GOLD layer:", gold_path + "/mental_health_clean.json")

# === 9. Print file sizes ===

def print_gcs_file_sizes(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        print(f"{blob.name} — {blob.size / (1024*1024):.2f} MB")

# Ejemplo:
bucket_name = "medallion-dat535"
prefix = "gold/"
print_gcs_file_sizes(bucket_name, prefix)