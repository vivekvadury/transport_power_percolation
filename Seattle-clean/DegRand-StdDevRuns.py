# %% [markdown]
# # System Vulnerabilities Analysis of Network #
# ## Targeted Attack vs Random Failure ##

# %% [markdown]
# Install Packages

# %%
import pandas as pd 
# import geopandas as gpd
import random
# import matplotlib.pyplot as plt
import networkx as nx
# import seaborn as sns 
import numpy as np
import networkit as nk

# %% [markdown]
# Read in Data

# %%
# Load node list from CSV
# Assuming the CSV has columns: 'node_id' and optionally other attributes
node_csv_path = "node_list_dc_btwn.csv"
nodes_df = pd.read_csv(node_csv_path)

# Load edge list from CSV
# Assuming the CSV has columns: 'source', 'target' and optionally other attributes
edge_csv_path = "edge_list.csv"
edges_df = pd.read_csv(edge_csv_path)

# %%
# Assuming your DataFrame is named 'df' and the column containing betweenness centrality is named 'betweenness_centrality'

# Find the maximum and minimum values
max_value = nodes_df['Betweenness Centrality'].max()
min_value = nodes_df['Betweenness Centrality'].min()
avg_value = nodes_df['Betweenness Centrality'].mean()

# Print the results
print(f"Maximum Betweenness Centrality: {max_value}")
print(f"Minimum Betweenness Centrality: {min_value}")
print(f"Average Betweenness Centrality: {avg_value}")

# %% [markdown]
# Build NetworkX Graph from NetworKit csv files

# %%
# Create an empty NetworkX graph
G = nx.Graph()

# %%
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
# Identify the largest connected component
largest_cc = max(nx.connected_components(G), key=len)

# Create a subgraph containing only the nodes in the largest connected component
G_sub_nx = G.subgraph(largest_cc).copy()  # Use `.copy()` to ensure it's a standalone graph

# Verify the node list remains the same
print("Number of nodes in NetworkX subgraph:", G_sub_nx.number_of_nodes())
print("Number of edges in NetworkX subgraph:", G_sub_nx.number_of_edges())

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

# %%
# plt.figure(figsize=(10, 6))
# plt.title(f'Targeted Attacks by Node Degree ({num_runs} Runs)')
# plt.xlabel('Fraction of nodes removed')
# plt.ylabel('Proportion of nodes in core')
# plt.plot(fraction_nodes_removed, targeted_attack_core_proportions_d_mean, marker='o', label='Mean', markersize=1, linewidth=1)
# plt.fill_between(fraction_nodes_removed, 
#                  targeted_attack_core_proportions_d_mean - targeted_attack_core_proportions_d_std,
#                  targeted_attack_core_proportions_d_mean + targeted_attack_core_proportions_d_std,
#                  alpha=0.2, label='± Std Dev')
# plt.legend()
# plt.grid(True)
# plt.show()

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

# %% [markdown]
# ## Random Attack

# %%
# Repeated Random Attack for confidence interval

N = G_sub_nx.number_of_nodes()
number_of_steps = N
M = N // number_of_steps

num_nodes_removed = range(0, G_sub_nx.number_of_nodes(), M)

# Run random attack multiple times
num_runs = 100
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


# # %%
# plt.figure(figsize=(10, 6))
# plt.title(f'Random Failure ({num_runs} Runs)')
# plt.xlabel('Fraction of nodes removed')
# plt.ylabel('Proportion of nodes in core')
# plt.plot(fraction_nodes_removed, random_attack_core_proportions_mean, marker='o', label='Mean', markersize=1, linewidth=1)
# plt.fill_between(fraction_nodes_removed, 
#                  random_attack_core_proportions_mean - random_attack_core_proportions_std,
#                  random_attack_core_proportions_mean + random_attack_core_proportions_std,
#                  alpha=0.2, label='± Std Dev')
# plt.legend()
# plt.grid(True)
# plt.show()

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


