"""
Page 3: Dockerized PySpark Cluster Architecture & Topology
"""

import os
import sys

dash_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dash_dir not in sys.path:
    sys.path.insert(0, dash_dir)

import streamlit as st
import pandas as pd
from utils import (
    inject_custom_css, render_sidebar, render_kpi_card,
    render_interpretation_box, render_viva_insight,
    render_mermaid_diagram
)

st.set_page_config(
    page_title="FMCG Spark Cluster Topology",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar()

st.markdown('<div class="main-header">Dockerized PySpark Cluster Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Distributed cluster topology, executor resource allocation, and parallel task scheduling mechanics</div>', unsafe_allow_html=True)

# Top KPI Section
ct1, ct2, ct3, ct4 = st.columns(4)
with ct1:
    render_kpi_card("Spark Master", "1 Master Node", "Port 7077 (RPC) • Web UI :8080", "#38BDF8")
with ct2:
    render_kpi_card("Spark Workers", "3 Worker Nodes", "spark-worker-1, 2, 3 Containers", "#10B981")
with ct3:
    render_kpi_card("Executor Cores", "6 Total Cores", "2 Cores Dedicated per Worker Node", "#F59E0B")
with ct4:
    render_kpi_card("Cluster Memory", "6 GB RAM", "2 GB Allocated per Worker Container", "#818CF8")

st.markdown("<br>", unsafe_allow_html=True)

# Cluster Container Topology Diagram (Full-Width Symmetrical Layout)
st.markdown("### 🖥️ Cluster Container Topology & Node Interconnect")
cluster_mermaid = """
graph TD
    classDef master fill:#1E293B,stroke:#38BDF8,stroke-width:2.5px,color:#F8FAFC;
    classDef worker fill:#0F172A,stroke:#10B981,stroke-width:2.5px,color:#F8FAFC;
    classDef driver fill:#312E81,stroke:#818CF8,stroke-width:2.5px,color:#F8FAFC;

    DRIVER["🚀 <b>PYSPARK DRIVER APPLICATION</b><br/><code>src/train_pyspark.py</code> (spark-submit)<br/>DAG Engine & Task Scheduler"]:::driver
    MASTER["⚡ <b>SPARK MASTER NODE (Standalone Coordinator)</b><br/>RPC Port: <b>7077</b> • Web UI: <b>http://localhost:8080</b><br/>Docker Container: <code>spark-master</code>"]:::master

    subgraph WORKERS ["🖥️ Distributed Cluster Worker Containers (Docker Network)"]
        W1["👷 <b>SPARK WORKER 1</b><br/><code>spark-worker-1</code><br/><b>2 Cores • 2.0 GB RAM</b><br/>Partitions: <b>1 to 5</b>"]:::worker
        W2["👷 <b>SPARK WORKER 2</b><br/><code>spark-worker-2</code><br/><b>2 Cores • 2.0 GB RAM</b><br/>Partitions: <b>6 to 10</b>"]:::worker
        W3["👷 <b>SPARK WORKER 3</b><br/><code>spark-worker-3</code><br/><b>2 Cores • 2.0 GB RAM</b><br/>Partitions: <b>11 to 14</b>"]:::worker
    end

    DRIVER -->|"Submit Distributed Job"| MASTER
    MASTER -->|"Assign Tasks 1-5"| W1
    MASTER -->|"Assign Tasks 6-10"| W2
    MASTER -->|"Assign Tasks 11-14"| W3
"""
render_mermaid_diagram(cluster_mermaid, height=520)

st.markdown("---")

# Distributed Execution Flow
st.markdown("### ⚡ Distributed Execution Flow & Task Scheduling")
flow_mermaid = """
graph LR
    classDef client fill:#312E81,stroke:#818CF8,stroke-width:2.5px,color:#F8FAFC;
    classDef dag fill:#1E293B,stroke:#38BDF8,stroke-width:2.5px,color:#F8FAFC;
    classDef task fill:#1E1B4B,stroke:#A855F7,stroke-width:2.5px,color:#F8FAFC;
    classDef exec fill:#064E3B,stroke:#34D399,stroke-width:2.5px,color:#F8FAFC;

    D1["🐍 <b>Python Driver</b><br/>Py4J IPC Gateway"]:::client
    D2["🗺️ <b>DAG Scheduler</b><br/>Stage Execution Graph"]:::dag
    D3["📋 <b>Task Scheduler</b><br/>14 Partition TaskSets"]:::task
    D4["⚙️ <b>Worker Executors</b><br/>Vectorized Tree Training"]:::exec

    D1 -->|"Transformations"| D2
    D2 -->|"Stage Graph"| D3
    D3 -->|"Parallel Tasks"| D4
"""
render_mermaid_diagram(flow_mermaid, height=240)

st.markdown("---")

# Core Distributed Concepts
st.subheader("⚙️ Key Distributed Computing Mechanisms in PySpark ML")

con_c1, con_c2 = st.columns(2)
with con_c1:
    st.markdown("""
    ##### 1. Partitioning & Data Locality
    - The 5,000,000 row dataset is partitioned into **14 discrete blocks**, ensuring each executor core handles ~350,000 records without overflowing L3 cache.
    - Tasks execute in-memory on the worker hosting the partition data to minimize cross-container network transport.
    
    ##### 2. Task Scheduling & Fault Tolerance
    - The Standalone Master dynamically assigns partition tasks to available worker slots. If an executor fails, Spark automatically replays the lineage DAG to reconstruct the partition.
    """)
with con_c2:
    st.markdown("""
    ##### 3. Shuffle & Partition Ensembling
    - Distributed tree algorithms (Random Forest) compute local histogram bins per partition and aggregate split boundaries across workers via Netty transport.
    - Partition-level ensembling for XGBoost/CatBoost enables workers to independently fit gradient trees and average predictions seamlessly across nodes.
    
    ##### 4. Memory Management & Garbage Collection
    - Configured with `spark.executor.memory = 1.5g` and `spark.memory.fraction = 0.8` to preserve Java heap stability during large matrix vectorization.
    """)

# Cluster Configuration Table
st.markdown("---")
st.subheader("📋 Active Cluster Hardware & Software Configuration")

config_data = [
    {"Parameter": "Spark Master URL", "Configured Value": "spark://spark-master:7077", "Description": "Standalone cluster coordinator"},
    {"Parameter": "Spark Version", "Configured Value": "Apache Spark 3.5.1 (Scala 2.12)", "Description": "Distributed compute framework"},
    {"Parameter": "Worker Containers", "Configured Value": "3 Workers (spark-worker-1, 2, 3)", "Description": "Isolated Docker container processes"},
    {"Parameter": "Cores per Worker", "Configured Value": "2 Cores (6 Total Executor Cores)", "Description": "Simultaneous parallel task execution capacity"},
    {"Parameter": "Memory per Worker", "Configured Value": "2.0 GB RAM (6.0 GB Total Cluster RAM)", "Description": "Worker host container memory ceiling"},
    {"Parameter": "Dataset Partitions", "Configured Value": "5 Partitions (1M) | 7 Partitions (3M, 5M)", "Description": "Parallel RDD / DataFrame splits"},
    {"Parameter": "Network Driver", "Configured Value": "Netty BlockTransferService (:41259)", "Description": "Inter-container high-speed data shuffle transport"},
    {"Parameter": "Virtualization", "Configured Value": "Docker Compose Bridge Network", "Description": "Containerized multi-node isolation"}
]
st.table(pd.DataFrame(config_data))

render_viva_insight(
    "Cluster Sizing & PySpark Worker Architecture",
    "In PySpark ML pipelines, Python processes communicate with JVM executor workers via IPC sockets (Py4J). Sizing workers with 2 cores and 2GB RAM prevents JVM garbage collection pauses while allowing PySpark workers to stream vectorized batches for tree training."
)
