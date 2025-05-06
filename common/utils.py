import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

def document_analysis(action, dataset_name, analyst=None):
    """
    Document analysis steps for forensic logging
    
    Parameters:
    -----------
    action: str
        Description of the analysis action performed
    dataset_name: str
        Name of the dataset being analyzed
    analyst: str, optional
        Name or ID of the person performing the analysis
        
    Returns:
    --------
    dict
        Log entry with timestamp and details
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "dataset": dataset_name,
        "analyst": analyst,
        "environment": {
            "platform": os.name,
            "python_version": platform.python_version(),
            "libraries": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": plt.matplotlib.__version__
            }
        }
    }
    
    return log_entry

def create_hash(file_path):
    """
    Create a hash of a file for integrity verification
    
    Parameters:
    -----------
    file_path: str
        Path to the file to hash
        
    Returns:
    --------
    str
        SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def create_lineage_graph(nodes, edges, title="Data Lineage"):
    """
    Create a visual graph representing data lineage
    
    Parameters:
    -----------
    nodes: list
        List of node dictionaries with 'id' and 'type' keys
    edges: list
        List of edge dictionaries with 'source', 'target', and 'label' keys
    title: str
        Title for the graph
        
    Returns:
    --------
    matplotlib figure
        Visualization of the lineage graph
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(node['id'], node_type=node.get('type', 'unknown'))
    
    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], label=edge.get('label', ''))
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Create layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    # Draw nodes with different colors based on type
    node_types = nx.get_node_attributes(G, 'node_type')
    node_colors = {'source': 'lightblue', 'derived': 'lightgreen', 
                  'processed': 'orange', 'final': 'pink', 'unknown': 'gray'}
    
    for node_type, color in node_colors.items():
        node_list = [node for node, type_val in node_types.items() if type_val == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, 
                              node_size=500, alpha=0.8, label=node_type)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def save_forensic_report(report, filename, include_timestamp=True):
    """
    Save forensic analysis report to file
    
    Parameters:
    -----------
    report: dict
        Report data structure
    filename: str
        Base filename to save to
    include_timestamp: bool
        Whether to append timestamp to filename
    
    Returns:
    --------
    str
        Path to saved file
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename, ext = os.path.splitext(filename)
        filename = f"{basename}_{timestamp}{ext}"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return filename

def compare_datasets(dataset1, dataset2, compare_type="columns"):
    """
    Compare two datasets to identify differences
    
    Parameters:
    -----------
    dataset1: pandas.DataFrame
        First dataset
    dataset2: pandas.DataFrame
        Second dataset
    compare_type: str
        Type of comparison - 'columns', 'statistics', or 'full'
        
    Returns:
    --------
    dict
        Comparison results
    """
    comparison = {
        "dataset1_shape": dataset1.shape,
        "dataset2_shape": dataset2.shape,
        "differences": {}
    }
    
    # Compare column structure
    if compare_type in ["columns", "full"]:
        cols1 = set(dataset1.columns)
        cols2 = set(dataset2.columns)
        
        comparison["differences"]["columns"] = {
            "only_in_dataset1": list(cols1 - cols2),
            "only_in_dataset2": list(cols2 - cols1),
            "common": list(cols1.intersection(cols2))
        }
    
    # Compare statistics for common columns
    if compare_type in ["statistics", "full"]:
        common_cols = set(dataset1.columns).intersection(set(dataset2.columns))
        stats_diff = {}
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(dataset1[col]) and pd.api.types.is_numeric_dtype(dataset2[col]):
                stats1 = dataset1[col].describe()
                stats2 = dataset2[col].describe()
                diff = {stat: float(stats1[stat] - stats2[stat]) for stat in stats1.index if stat in stats2.index}
                stats_diff[col] = diff
        
        comparison["differences"]["statistics"] = stats_diff
    
    # Full record-level comparison (for smaller datasets)
    if compare_type == "full" and dataset1.shape[0] < 10000 and dataset2.shape[0] < 10000:
        try:
            # Attempt to find exact matching records
            merged = pd.merge(dataset1, dataset2, indicator=True, how='outer')
            only_in_1 = merged[merged['_merge'] == 'left_only'].shape[0]
            only_in_2 = merged[merged['_merge'] == 'right_only'].shape[0]
            common = merged[merged['_merge'] == 'both'].shape[0]
            
            comparison["differences"]["records"] = {
                "only_in_dataset1": only_in_1,
                "only_in_dataset2": only_in_2,
                "common": common
            }
        except:
            comparison["differences"]["records"] = "Could not perform record-level comparison"
    
    return comparison