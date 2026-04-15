# %% [markdown]
# # System Vulnerabilities Analysis of Network #
# ## Targeted Attack vs Random Failure ##

# %% [markdown]
# Install Packages

# %%
import pandas as pd 
import geopandas as gpd
import random
# %pip install matplotlib 
import matplotlib.pyplot as plt
import networkx as nx
# %pip install seaborn
import seaborn as sns 
import numpy as np
import networkit as nk

# %%
# Setup output logging to file
import sys
from datetime import datetime
from io import StringIO

class OutputLogger:
    def __init__(self, filename='notebook_outputs.txt'):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = StringIO()
        
        # Write header
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NOTEBOOK OUTPUT LOG\n")
            f.write(f"Notebook: 2-NetworkX-Target-approxBC-v-Random.py\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # Append to file in real-time
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()

# Initialize output logger
output_logger = OutputLogger('notebook_outputs.txt')
sys.stdout = output_logger

print("✓ Output logging initialized - all outputs will be saved to notebook_outputs.txt")

# %%
# Cell runtime tracking helper
import time

class CellTimer:
    def __init__(self):
        self.start_time = None
    
    def start(self, cell_name=""):
        self.start_time = time.time()
        self.cell_name = cell_name
    
    def end(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"⏱ Cell runtime: {elapsed:.3f} seconds")
            self.start_time = None

# Initialize cell timer
cell_timer = CellTimer()

# %% [markdown]
# Read in Data

# %%
cell_timer.start()
# Load node list from CSV
# Assuming the CSV has columns: 'node_id' and optionally other attributes
node_csv_path = "node_list_dc_btwn.csv"
nodes_df = pd.read_csv(node_csv_path)

# Load edge list from CSV
# Assuming the CSV has columns: 'source', 'target' and optionally other attributes
edge_csv_path = "edge_list.csv"
edges_df = pd.read_csv(edge_csv_path)
cell_timer.end()

# %%
cell_timer.start()
# Assuming your DataFrame is named 'df' and the column containing betweenness centrality is named 'betweenness_centrality'

# Find the maximum and minimum values
max_value = nodes_df['Betweenness Centrality'].max()
min_value = nodes_df['Betweenness Centrality'].min()
avg_value = nodes_df['Betweenness Centrality'].mean()

# Print the results
print(f"Maximum Betweenness Centrality: {max_value}")
print(f"Minimum Betweenness Centrality: {min_value}")

print(f"Average Betweenness Centrality: {avg_value}")
cell_timer.end()

# %% [markdown]
# Build NetworkX Graph from NetworKit csv files

# %%
# Create an empty NetworkX graph
G = nx.Graph()

# %%
cell_timer.start()
# Add nodes with attributes
for _, row in nodes_df.iterrows():
    node_id = row['Node ID']
    attributes = row.drop('Node ID').to_dict()  # Convert other columns to attributes
    G.add_node(node_id, **attributes)

# Add edges with attributes
for _, row in edges_df.iterrows():
    source = row['Source']
    target = row['Target']
    attributes = row.drop(['Source', 'Target']).to_dict()  # Convert other columns to attributes
    G.add_edge(source, target, **attributes)

# Verify the graph is created
print("Number of nodes in NetworkX graph:", G.number_of_nodes())

print("Number of edges in NetworkX graph:", G.number_of_edges())
cell_timer.end()

# %%
# Assuming 'G_sub_nx' is your NetworkX graph and betweenness centrality is already calculated as a node attribute

# Find the node with the maximum betweenness centrality
max_node = max(G.nodes, key=lambda node: G.nodes[node].get('Betweenness Centrality', 0))
max_value = G.nodes[max_node].get('Betweenness Centrality', 0)

# Print the results
print(f"Node with maximum betweenness centrality: {max_node}")
print(f"Maximum betweenness centrality value: {max_value}")

# %%
print(attributes.keys())

# %%
# Convert DataFrame to a dictionary of dictionaries
attributes = nodes_df.set_index('Node ID').to_dict('index')

# Assign attributes to the graph
nx.set_node_attributes(G, attributes)

# %%
# Create a DataFrame from the NetworkX graph
nodes_df = pd.DataFrame(G.nodes(data=True), columns=['Node ID', 'Attributes'])
edges_df = pd.DataFrame(G.edges(data=True), columns=['Source', 'Target', 'Attributes'])

# Save the node and edge DataFrames to CSV
nodes_df.to_csv("nodes_nx.csv", index=False)
edges_df.to_csv("edges_nx.csv", index=False)

# %% [markdown]
# ## Identify Largest Connected Component for Analysis

# %%
cell_timer.start()
# Identify the largest connected component
largest_cc = max(nx.connected_components(G), key=len)

# Create a subgraph containing only the nodes in the largest connected component
G_sub_nx = G.subgraph(largest_cc).copy()  # Use `.copy()` to ensure it's a standalone graph

# Verify the node list remains the same
print("Number of nodes in NetworkX subgraph:", G_sub_nx.number_of_nodes())

print("Number of edges in NetworkX subgraph:", G_sub_nx.number_of_edges())
cell_timer.end()

# %% [markdown]
# Compare Metrics

# %%
print("Total Number of Edges in Network: ",G.number_of_edges())
print("Number of Edges in Subnetwork: ",G_sub_nx.number_of_edges())

# %%
print("Total Number of Nodes in Network: ",G.number_of_nodes())
print("Number of Nodes in Subnetwork: ",G_sub_nx.number_of_nodes())

# %%
print("Average degree of Network = ",2*G.number_of_edges()/G.number_of_nodes())
print("Average degree of Subnetwork = ",2*G_sub_nx.number_of_edges()/G_sub_nx.number_of_nodes())

# %%
# Compute clustering coefficient
clustering_coeffs = nx.clustering(G_sub_nx)
min_clustering = min(clustering_coeffs.values())
max_clustering = max(clustering_coeffs.values())
print(f"Minimum clustering coefficient: {min_clustering}")
print(f"Maximum clustering coefficient: {max_clustering}")
print("Average Clustering Coefficient of Network = ", nx.average_clustering(G))
print("Average Clustering Coefficient of Subnetwork = ", nx.average_clustering(G_sub_nx))

# %%
# Compute degree
degrees = dict(G_sub_nx.degree())
min_degree = min(degrees.values())
max_degree = max(degrees.values())
print(f"Minimum degree: {min_degree}")
print(f"Maximum degree: {max_degree}")

# %%
# Assuming 'G_sub_nx' is your NetworkX graph and betweenness centrality is already calculated as a node attribute

# Find the node with the maximum betweenness centrality
max_node = max(G_sub_nx.nodes, key=lambda node: G_sub_nx.nodes[node].get('Betweenness Centrality', 0))
max_value = G_sub_nx.nodes[max_node].get('Betweenness Centrality', 0)

# Print the results
print(f"Node with maximum betweenness centrality: {max_node}")
print(f"Maximum betweenness centrality value: {max_value}")

# %%
# Create a DataFrame from the subgraph
nodes_df_sub = pd.DataFrame(G_sub_nx.nodes(data=True), columns=['Node ID', 'Attributes'])
edges_df_sub = pd.DataFrame(G_sub_nx.edges(data=True), columns=['Source', 'Target', 'Attributes'])

# Save the subgraph node and edge DataFrames to CSV
nodes_df_sub.to_csv("nodes_nx_sub.csv", index=False)
edges_df_sub.to_csv("edges_nx_sub.csv", index=False)

# %% [markdown]
# # Targetted Attacks ##

# %% [markdown]
# Nodes sorted by Degree for Subnetwork

# %%
cell_timer.start()
N = G_sub_nx.number_of_nodes()
number_of_steps = N
M = N // number_of_steps

num_nodes_removed = range(0, N, M)

# Run targeted attack by degree multiple times
num_runs = 50
all_degree_attack_runs = []
fraction_nodes_removed = []

for run in range(num_runs):
    C = G_sub_nx.copy()
    targeted_attack_core_proportions_d = []
    
    for nodes_removed in num_nodes_removed:
        # Measure the relative size of the network core
        core = max(nx.connected_components(C), key=len)
        
        core_proportion = len(core) / N
        targeted_attack_core_proportions_d.append(core_proportion)
        
        # Only add to fraction_nodes_removed once (same for all runs)
        if run == 0:
            fraction_nodes_removed.append(nodes_removed / N)

        # If there are more than M nodes, select top M nodes and remove them
        if C.number_of_nodes() > M:
            # Sort by degree descending
            deg_list = sorted([(C.degree(n), n) for n in C.nodes], reverse=True)
            # Degree at the cut line
            cutoff_degree = deg_list[M - 1][0] if len(deg_list) >= M else deg_list[-1][0]
            # Everyone strictly above cutoff is guaranteed
            guaranteed = [node for deg, node in deg_list if deg > cutoff_degree]
            # Ties at the cutoff degree compete randomly
            tied_candidates = [node for deg, node in deg_list if deg == cutoff_degree]
            remaining = M - len(guaranteed)
            sampled = random.sample(tied_candidates, min(remaining, len(tied_candidates))) if remaining > 0 else []
            nodes_to_remove = guaranteed + sampled
            C.remove_nodes_from(nodes_to_remove)
    
    all_degree_attack_runs.append(targeted_attack_core_proportions_d)

# Calculate mean and standard deviation across all runs
targeted_attack_core_proportions_d_mean = np.mean(all_degree_attack_runs, axis=0)
targeted_attack_core_proportions_d_std = np.std(all_degree_attack_runs, axis=0)

print(f"Completed {num_runs} degree-based targeted attack simulations")
print(f"Mean core proportion: {targeted_attack_core_proportions_d_mean}")

# Keep the original for backward compatibility
targeted_attack_core_proportions_d = targeted_attack_core_proportions_d_mean

cell_timer.end()

# %%
# Save results to CSV file
# Calculate nodes removed from fraction
nodes_removed_list = [int(frac * N) for frac in fraction_nodes_removed]

results_deg_df = pd.DataFrame({
    'Nodes Removed': nodes_removed_list,
    'Fraction of Nodes Removed': fraction_nodes_removed,
    'Mean Core Proportion': targeted_attack_core_proportions_d_mean,
    'Std Dev Core Proportion': targeted_attack_core_proportions_d_std
})
results_deg_df.to_csv('TA-Degree.csv', index=False)

# %%
# Print value of M
print("Value of M:", M)

# %%
# Calculate the number of nodes removed in first 3 steps
num_nodes_removed_first_3_steps = 3 * M
print("Number of nodes removed in first 3 steps:", num_nodes_removed_first_3_steps)

# %% [markdown]
# ### Nodes sorted by Betweenness Centrality for Subnetwork ###
# Static Prioritization: Betweenness Centrality ranked only at beginning, no dynamic updates

# %%
cell_timer.start()
# Adopted code from Marta
# Assuming betweenness centrality values are stored in the `attributes` dictionary
N = G_sub_nx.number_of_nodes()
number_of_steps = N
M = N // number_of_steps

num_nodes_removed = range(0, N, M)
C = G_sub_nx.copy()
targeted_attack_core_proportions_btwn = []
fraction_nodes_removed = []
nodes_removed_list = []

############# Check the copy
# Convert nodes to DataFrame
nodes_df_sub = pd.DataFrame(C.nodes(data=True), columns=['Node ID', 'Attributes'])

# Expand attribute dictionary into separate columns
nodes_df_sub = nodes_df_sub.join(pd.json_normalize(nodes_df_sub.pop("Attributes")))

# Sort by "Betweenness Centrality" in descending order
nodes_df_sub = nodes_df_sub.sort_values(by="Betweenness Centrality", ascending=False)

# # Save to CSV
nodes_df_sub.to_csv("copy_nodes_nx_sub.csv", index=False)

# Convert edges to DataFrame (unchanged)
edges_df_sub = pd.DataFrame(C.edges(data=True), columns=['Source', 'Target', 'Attributes'])
edges_df_sub = edges_df_sub.join(pd.json_normalize(edges_df_sub.pop("Attributes")))
edges_df_sub.to_csv("copy_edges_nx_sub.csv", index=False)

for nodes_removed in num_nodes_removed:
    # Measure the relative size of the network core
    core = max(nx.connected_components(C), key=len)
    core_proportion = len(core) / N

    # Save the largest connected component (core) nodes and edges to CSV
    C_core = C.subgraph(core).copy()  # Create a subgraph of the largest connected component

    # Save core nodes to CSV
    core_nodes_df = pd.DataFrame(C_core.nodes(data=True), columns=['Node ID', 'Attributes'])
    core_nodes_df = core_nodes_df.join(pd.json_normalize(core_nodes_df.pop("Attributes")))

    # Calculate log of betweenness centrality
    if 'Betweenness Centrality' in core_nodes_df.columns:
        core_nodes_df['Log Betweenness Centrality'] = np.log(core_nodes_df['Betweenness Centrality'].replace(0, np.nan))

    core_nodes_df.to_csv(f"TA_Core_Nodes/core_nodes_{nodes_removed}.csv", index=False)

    # Save core edges to CSV
    core_edges_df = pd.DataFrame(C_core.edges(data=True), columns=['Source', 'Target', 'Attributes'])
    core_edges_df = core_edges_df.join(pd.json_normalize(core_edges_df.pop("Attributes")))
    core_edges_df.to_csv(f"TA_Core_Edges/core_edges_{nodes_removed}.csv", index=False)

    targeted_attack_core_proportions_btwn.append(core_proportion)
    fraction_nodes_removed.append(nodes_removed / N)
    nodes_removed_list.append(nodes_removed)
    # print(nodes_removed,core_proportion)

    # If there are more than M nodes, select top M nodes by betweenness centrality and remove them
    if C.number_of_nodes() > M:
        # Ensure node IDs match the keys in the attributes dictionary
        nodes_sorted_by_betweenness = sorted(
            C.nodes, 
            key=lambda node: attributes[int(node)]['Betweenness Centrality'],  # Convert node ID to int if necessary
            reverse=True
        )
        nodes_to_remove = nodes_sorted_by_betweenness[:M]
        C.remove_nodes_from(nodes_to_remove)        

    # Save removed nodes and their attributes to a csv file
        removed_nodes_data = []
        # Create a DataFrame with Node ID, Betweenness Centrality, Coordinates, and Step
        for node in nodes_to_remove:
            removed_nodes_data.append({
                "Node ID": node,
                "Betweenness Centrality": attributes[int(node)]['Betweenness Centrality'],
                "Longitude": attributes[int(node)]['Longitude'],
                "Latitude": attributes[int(node)]['Latitude'],
                "Step": nodes_removed
            })
        removed_nodes_df = pd.DataFrame(removed_nodes_data)
        removed_nodes_df.to_csv(f"TargetAttack_Removal/TA_removed_nodes_{nodes_removed}.csv", index=False)

# Save the results to a CSV file
results_btwn_df = pd.DataFrame({
    'Nodes Removed': nodes_removed_list,
    'Fraction of Nodes Removed': fraction_nodes_removed,
    'Core Proportion': targeted_attack_core_proportions_btwn
})
results_btwn_df.to_csv('TA-BC-Static.csv', index=False)


print("completed targeted attack by betweenness centrality (static)")

cell_timer.end()

# %% [markdown]
# ### Nodes sorted by Betweenness Centrality for Subnetwork ###
# Dynamic Prioritization: Betweenness Centrality Recomputed after Each Node Removal Step

# %%
cell_timer.start()
# Adopted code from Marta
# Recompute betweenness centrality of remaining largest connected component after each node removal
N = G_sub_nx.number_of_nodes()
number_of_steps = N
M = N // number_of_steps

num_nodes_removed = range(0, N, M)
C = G_sub_nx.copy()
targeted_attack_core_proportions_btwn_dyn = []
fraction_nodes_removed = []
nodes_removed_list = []

############# Check the copy
# Convert nodes to DataFrame
nodes_df_sub = pd.DataFrame(C.nodes(data=True), columns=['Node ID', 'Attributes'])

# Expand attribute dictionary into separate columns
nodes_df_sub = nodes_df_sub.join(pd.json_normalize(nodes_df_sub.pop("Attributes")))

# Sort by "Betweenness Centrality" in descending order
nodes_df_sub = nodes_df_sub.sort_values(by="Betweenness Centrality", ascending=False)

# Save to CSV
nodes_df_sub.to_csv("copy_nodes_nx_sub_dyn.csv", index=False)

# Convert edges to DataFrame (unchanged)
edges_df_sub = pd.DataFrame(C.edges(data=True), columns=['Source', 'Target', 'Attributes'])
edges_df_sub = edges_df_sub.join(pd.json_normalize(edges_df_sub.pop("Attributes")))
edges_df_sub.to_csv("copy_edges_nx_sub_dyn.csv", index=False)

for nodes_removed in num_nodes_removed:
    # Measure the relative size of the network core
    core = max(nx.connected_components(C), key=len)
    core_proportion = len(core) / N

    # Save the largest connected component (core) nodes and edges to CSV
    C_core = C.subgraph(core).copy()  # Create a subgraph of the largest connected component

    # Save core nodes to CSV
    core_nodes_df = pd.DataFrame(C_core.nodes(data=True), columns=['Node ID', 'Attributes'])
    core_nodes_df = core_nodes_df.join(pd.json_normalize(core_nodes_df.pop("Attributes")))

    # Calculate log of betweenness centrality
    if 'Betweenness Centrality' in core_nodes_df.columns:
        core_nodes_df['Log Betweenness Centrality'] = np.log(core_nodes_df['Betweenness Centrality'].replace(0, np.nan))

    core_nodes_df.to_csv(f"TA_Core_Nodes-Dynamic/core_nodes_{nodes_removed}.csv", index=False)

    # Save core edges to CSV
    core_edges_df = pd.DataFrame(C_core.edges(data=True), columns=['Source', 'Target', 'Attributes'])
    core_edges_df = core_edges_df.join(pd.json_normalize(core_edges_df.pop("Attributes")))
    core_edges_df.to_csv(f"TA_Core_Edges-Dynamic/core_edges_{nodes_removed}.csv", index=False)

    targeted_attack_core_proportions_btwn_dyn.append(core_proportion)
    fraction_nodes_removed.append(nodes_removed / N)
    nodes_removed_list.append(nodes_removed)
    # print(nodes_removed,core_proportion)

    # If there are more than M nodes, recompute betweenness centrality and select top M nodes to remove
    if C.number_of_nodes() > M:
        # Convert current graph from NetworkX to NetworKit
        C_nk = nk.nxadapter.nx2nk(C)
        
        # Recompute approximate betweenness centrality for the current graph
        btwn_approx = nk.centrality.ApproxBetweenness(C_nk, epsilon=0.01)
        btwn_approx.run()
        current_bc_scores = btwn_approx.scores()
        
        # Align scores back to original NetworkX node labels using current ordering
        node_order = list(C.nodes())
        current_bc = dict(zip(node_order, current_bc_scores))
        
        # Get nodes sorted by updated betweenness centrality in descending order
        nodes_sorted_by_betweenness = sorted(
            C.nodes, 
            key=lambda node: current_bc[node],
            reverse=True
        )
        nodes_to_remove = nodes_sorted_by_betweenness[:M]
        C.remove_nodes_from(nodes_to_remove)        

    # Save removed nodes and their attributes to a csv file
        removed_nodes_data = []
        # Create a DataFrame with Node ID, Betweenness Centrality, Coordinates, and Step
        for node in nodes_to_remove:
            removed_nodes_data.append({
                "Node ID": node,
                "Betweenness Centrality": current_bc[node],
                "Longitude": attributes[int(node)]['Longitude'],
                "Latitude": attributes[int(node)]['Latitude'],
                "Step": nodes_removed
            })
        removed_nodes_df = pd.DataFrame(removed_nodes_data)
        removed_nodes_df.to_csv(f"TargetAttack_Removal-Dynamic/TA_removed_nodes_{nodes_removed}.csv", index=False)

# Save the results to a CSV file
results_btwn_dyn_df = pd.DataFrame({
    'Nodes Removed': nodes_removed_list,
    'Fraction of Nodes Removed': fraction_nodes_removed,
    'Core Proportion': targeted_attack_core_proportions_btwn_dyn
})
results_btwn_dyn_df.to_csv('TA-BC-Dynamic.csv', index=False)


print("completed targeted attacks by betweenness centrality (dynamic)")
cell_timer.end()

# %% [markdown]
# ## Random Attack

# %%
import random

# %%
cell_timer.start()
# Repeated Random Attack for confidence interval

N = G_sub_nx.number_of_nodes()
number_of_steps = N
M = N // number_of_steps

num_nodes_removed = range(0, G_sub_nx.number_of_nodes(), M)

# Run random attack multiple times
num_runs = 50
all_random_attack_runs = []
fraction_nodes_removed = []

for run in range(num_runs):
    C = G_sub_nx.copy()
    random_attack_core_proportions = []
    
    for nodes_removed in num_nodes_removed:
        # Measure the relative size of the network core
        core = max(nx.connected_components(C), key=len)
        core_proportion = len(core) / N
        random_attack_core_proportions.append(core_proportion)
        
        # Only add to fraction_nodes_removed once (same for all runs)
        if run == 0:
            fraction_nodes_removed.append(nodes_removed / N)

        # If there are more than M nodes, select M nodes at random and remove them
        if C.number_of_nodes() > M:
            nodes_to_remove = random.sample(list(C.nodes), M)
            C.remove_nodes_from(nodes_to_remove)
    
    all_random_attack_runs.append(random_attack_core_proportions)

# Calculate mean and standard deviation across all runs
random_attack_core_proportions_mean = np.mean(all_random_attack_runs, axis=0)
random_attack_core_proportions_std = np.std(all_random_attack_runs, axis=0)

print(f"Completed {num_runs} random attack simulations")
print(f"Mean core proportion: {random_attack_core_proportions_mean}")

cell_timer.end()


# %%
# Save plot data to CSV
# Calculate nodes removed from fraction
N = G_sub_nx.number_of_nodes()
nodes_removed_list = [int(frac * N) for frac in fraction_nodes_removed]

random_attack_results_df = pd.DataFrame({
    'Nodes Removed': nodes_removed_list,
    'Fraction of Nodes Removed': fraction_nodes_removed,
    'Mean Core Proportion': random_attack_core_proportions_mean,
    'Std Dev Core Proportion': random_attack_core_proportions_std
})
random_attack_results_df.to_csv('RA-Results.csv', index=False)

# %% [markdown]
# ## Close Output Logger

# %%
# Restore stdout and finalize output log
sys.stdout = output_logger.terminal

with open('notebook_outputs.txt', 'a', encoding='utf-8') as f:
    f.write("\n" + "="*80 + "\n")
    f.write("END OF NOTEBOOK OUTPUT LOG\n")
    f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")

print("✓ Output logging completed - all outputs saved to notebook_outputs.txt")


