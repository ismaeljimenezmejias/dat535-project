from pyspark import SparkContext
from pyspark.sql import SparkSession
import csv
import io

# === 0. Spark ===
sc = SparkContext(appName="MentalHealthCleaningRDD")
spark = SparkSession.builder.appName("MentalHealthCleaningRDD").getOrCreate()

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
total_rows = rdd_merged.count()
print(f"Total cleaned rows: {total_rows}")

# === 7. Convert dict -> DataFrame ===
df_clean = rdd_merged.toDF()

# === 8. Repartition para archivos más manejables ===
num_partitions = 10  # ajusta según tamaño de dataset y memoria
df_clean = df_clean.repartition(num_partitions)

# === 9. Save cleaned dataset to GOLD layer ===
gold_path = "gs://medallion-dat535/gold"

# Parquet
df_clean.write.mode("overwrite").parquet(f"{gold_path}/mental_health_clean.parquet")
print("Saved Parquet dataset to GOLD layer.")

# CSV
df_clean.write.mode("overwrite").option("header", True).csv(f"{gold_path}/mental_health_clean.csv")
print("Saved CSV dataset to GOLD layer.")

# JSON
df_clean.write.mode("overwrite").json(f"{gold_path}/mental_health_clean.json")
print("Saved JSON dataset to GOLD layer.")
