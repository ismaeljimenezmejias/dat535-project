from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import csv
import io
import time
import pandas as pd
from google.cloud import storage

# === 0. Spark ===
sc = SparkContext(appName="MentalHealthCleaning")
spark = SparkSession.builder.appName("MentalHealthCleaning").getOrCreate()

silver_path = "gs://medallion-dat535/silver/mental_health_structured.csv"
gold_path = "gs://medallion-dat535/gold"

# === 1. RDD approach ===
start_rdd = time.time()

rdd = sc.textFile(silver_path)
header = rdd.first()
columns = header.split(",")
rdd_rows = rdd.filter(lambda x: x != header)

def parse_line(line):
    reader = csv.reader(io.StringIO(line))
    row = next(reader)
    return dict(zip(columns, row))

rdd_dict = rdd_rows.map(parse_line)

def no_nulls(record):
    return all(v not in ("", None, "NULL") for v in record.values())

rdd_nonull = rdd_dict.filter(no_nulls)

def norm(record):
    return {k: (v.title() if isinstance(v, str) else v) for k, v in record.items()}

rdd_norm = rdd_nonull.map(norm)

def merge_sw(r):
    if ("SocialWeaknessPrimary" in r and
        "SocialWeaknessSecondary" in r and
        r["SocialWeaknessPrimary"] == r["SocialWeaknessSecondary"]):
        r["SocialWeakness"] = r["SocialWeaknessPrimary"]
        del r["SocialWeaknessPrimary"]
        del r["SocialWeaknessSecondary"]
    return r

rdd_merged = rdd_norm.map(merge_sw)
total_rows = rdd_merged.count()
print(f"RDD cleaned rows: {total_rows}")

metadata_cols = ["_ingestion_timestamp", "_source", "_status", "_raw_data"]

def drop_metadata(record):
    return {k: v for k, v in record.items() if k not in metadata_cols}

rdd_no_meta = rdd_merged.map(drop_metadata)

df_rdd_clean = rdd_no_meta.toDF().repartition(10)

# Guardar en GOLD con carpeta propia
df_rdd_clean.write.mode("overwrite").parquet(f"{gold_path}/rdd/mental_health_clean.parquet")
df_rdd_clean.write.mode("overwrite").option("header", True).csv(f"{gold_path}/rdd/mental_health_clean.csv")
df_rdd_clean.write.mode("overwrite").json(f"{gold_path}/rdd/mental_health_clean.json")

print(f"✅ RDD cleaning completed in {time.time() - start_rdd:.2f} seconds.\n")

# === Check SIZES of RDD outputs in GCS ===
from google.cloud import storage

# Inicializa cliente GCS
client = storage.Client()
bucket_name = "medallion-dat535"
bucket = client.bucket(bucket_name)

# Carpeta GOLD donde guardaste los resultados de RDD
rdd_folders = [
    "gold/rdd/mental_health_clean.parquet/",
    "gold/rdd/mental_health_clean.csv/",
    "gold/rdd/mental_health_clean.json/"
]

def get_folder_size(bucket, folder):
    """Devuelve tamaño total de todos los objetos en la carpeta en bytes."""
    total_size = 0
    blobs = client.list_blobs(bucket, prefix=folder)
    for blob in blobs:
        total_size += blob.size
    return total_size

for folder in rdd_folders:
    size_bytes = get_folder_size(bucket, folder)
    size_mb = size_bytes / (1024*1024)
    print(f"{folder}: {size_bytes} bytes ({size_mb:.2f} MB)")


# === 2. Spark DataFrame approach ===
start_df = time.time()

df = spark.read.option("header", True).csv(silver_path)

metadata_cols = ["_ingestion_timestamp", "_source", "_status", "_raw_data"]
existing_metadata_cols = [c for c in metadata_cols if c in df.columns]
if existing_metadata_cols:
    df = df.drop(*existing_metadata_cols)

for c in df.columns:
    df = df.filter((col(c).isNotNull()) & (col(c) != "") & (col(c) != "NULL"))

text_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType"]
for c in text_cols:
    df = df.withColumn(c, col(c).substr(1,1).upper() + col(c).substr(2, 1000))

if "SocialWeaknessPrimary" in df.columns and "SocialWeaknessSecondary" in df.columns:
    df = df.withColumn(
        "SocialWeakness",
        when(col("SocialWeaknessPrimary") == col("SocialWeaknessSecondary"), col("SocialWeaknessPrimary"))
    ).drop("SocialWeaknessPrimary", "SocialWeaknessSecondary")

total_rows = df.count()
print(f"Spark DF cleaned rows: {total_rows}")

df.write.mode("overwrite").parquet(f"{gold_path}/spark_df/mental_health_clean.parquet")
df.write.mode("overwrite").option("header", True).csv(f"{gold_path}/spark_df/mental_health_clean.csv")
df.write.mode("overwrite").json(f"{gold_path}/spark_df/mental_health_clean.json")

print(f"✅ Spark DataFrame cleaning completed in {time.time() - start_df:.2f} seconds.\n")

# === 3. pandas approach ===
start_pd = time.time()

client = storage.Client()
bucket = client.bucket("medallion-dat535")
blob = bucket.blob("silver/mental_health_structured.csv")
blob.download_to_filename("mental_health_structured.csv")

df_pd = pd.read_csv("mental_health_structured.csv")
df_pd_clean = df_pd.dropna()
metadata_cols_pd = ["_ingestion_timestamp", "_source", "_status", "_raw_data"]
df_pd_clean = df_pd_clean.drop(columns=[c for c in metadata_cols_pd if c in df_pd_clean.columns])

text_cols_pd = df_pd_clean.select_dtypes(include="object").columns
df_pd_clean[text_cols_pd] = df_pd_clean[text_cols_pd].apply(lambda x: x.str.title())

if "SocialWeaknessSecondary" in df_pd_clean.columns:
    if (df_pd_clean["SocialWeaknessPrimary"] == df_pd_clean["SocialWeaknessSecondary"]).all():
        df_pd_clean = df_pd_clean.drop(columns=["SocialWeaknessSecondary"])
        df_pd_clean = df_pd_clean.rename(columns={"SocialWeaknessPrimary":"SocialWeakness"})

df_pd_clean.to_csv("mental_health_clean_pandas.csv", index=False, encoding="utf-8")
blob_out = bucket.blob("gold/pandas/mental_health_clean_pandas.csv")
blob_out.upload_from_filename("mental_health_clean_pandas.csv")

print(f"✅ pandas cleaning completed in {time.time() - start_pd:.2f} seconds.\n")
