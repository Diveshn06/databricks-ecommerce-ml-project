# End-to-End E-Commerce Customer Segmentation using Databricks

## Project Overview

This project demonstrates how to build an end-to-end Data & AI pipeline using Databricks. The objective is to identify high-value customers from an e-commerce dataset by building a Medallion Architecture pipeline and training a machine learning model.

The solution processes raw transaction data, performs feature engineering, and predicts whether a customer is a high-value customer based on purchasing behavior.

---

## Problem Statement

Businesses often struggle to identify their most valuable customers. By analyzing transaction history, we can build a model that classifies customers based on their purchasing behavior.

In this project, customers who spend more than a defined threshold are classified as **High Value Customers**.

---

## Dataset

Online Retail dataset containing transaction records including:

* Invoice Number
* Product Code
* Product Description
* Quantity
* Invoice Date
* Unit Price
* Customer ID
* Country

---

## Architecture

Medallion Architecture implemented in Databricks:

Raw Dataset
↓
Bronze Layer – Raw transactional data
↓
Silver Layer – Cleaned and transformed data
↓
Gold Layer – Customer level aggregated features
↓
Machine Learning Model – Random Forest Classifier

---

## Data Pipeline

### Bronze Layer

Stores raw transactional data.

Table:
bronze_ecommerce

### Silver Layer

Data cleaning and transformation:

* Removed null CustomerID values
* Created TotalPrice column

Table:
silver_ecommerce

### Gold Layer

Customer level aggregation:

* Total Orders
* Total Spending
* Average Order Value

Table:
gold_customer_features

---

## Machine Learning Model

A Random Forest classifier was trained to identify high value customers using the following features:

* total_orders
* total_spent
* avg_order_value

Label definition:
High value customers are defined as customers whose total spending exceeds a defined threshold.

---

## Model Evaluation

Accuracy Score: **1.0**

---

## Tools & Technologies

* Databricks
* PySpark
* Delta Lake
* Python
* Scikit-learn
* MLflow

---

## Key Learnings

* Built a complete Medallion Architecture pipeline
* Performed feature engineering using PySpark
* Implemented a machine learning model
* Tracked experiments using MLflow
* Designed an end-to-end data & AI workflow

---

## Author

Divesh Negi
