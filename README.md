# DAT535 – Mental Health Pipeline  
### Bronze–Silver–Gold Pipeline in Apache Spark (Google Cloud Dataproc)

This repository contains the full implementation of our DAT535 project, including
Python scripts, Jupyter Notebooks, the final report (PDF), and required datasets.

## Project Overview
The goal of this project is to implement a Medallion-Architecture data pipeline  
(Bronze → Silver → Gold) on an intentionally unstructured mental-health dataset.  
All computation was executed on a Google Cloud Dataproc Spark cluster.

The pipeline includes:

- **Bronze layer (Ingestion)**  
  - Load unstructured JSON-like raw data  
  - Retain malformed and inconsistent records  
  - Add ingestion timestamp  
  - Store results as Parquet

- **Silver layer (Cleaning)**  
  - Missing-value normalization  
  - Capitalization standardization  
  - Flattening of nested JSON-like fields  
  - Duplicate removal  
  - Merging semantically identical columns  
  - Output: clean, consistent schema

- **Gold layer (Serving / Analytics)**  
  - Aggregations, statistics and transformations  
  - Comparison of RDD vs DataFrame vs SQL execution  
  - Outputs used for evaluation and report figures

---

## Repository Structure

