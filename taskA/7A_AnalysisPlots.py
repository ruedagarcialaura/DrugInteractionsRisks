import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os
import argparse

def generate_formal_plot(csv_path):
    # Set Seaborn style for a clean, professional look
    sns.set_theme(style="white")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Extract metadata from input filename for better plot naming
    input_filename = os.path.basename(csv_path)
    
    if "APRIORI" in input_filename.upper():
        algo_name = "A Priori"
    elif "FP_GROWTH" in input_filename.upper() or "FP-GROWTH" in input_filename.upper():
        algo_name = "FP-Growth"
    else:
        algo_name = "Association Rules"

    if "drug_names" in input_filename:
        item_type = "Drug Names"
    elif "active_substances" in input_filename:
        item_type = "Active Substances"
    else:
        item_type = "Items"

    try:
        support_part = input_filename.split('_')[-1].replace('.csv', '')
        support_val = support_part.replace('_', '.')
        support_text = f"Min. Support: {support_val}"
    except:
        support_text = "Min. Support: N/A"



    df = pd.read_csv(csv_path)
    # We take the top rules to avoid cluttering the professional look
    top_rules = df.sort_values('lift', ascending=False).head(12)

    G = nx.DiGraph()
    reactions_list = set()

    for _, row in top_rules.iterrows():
        ant = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
        con = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "")
        G.add_edge(ant, con, weight=row['lift'])
        reactions_list.add(con)

    # Calculate positions using a spring layout with high repulsion
    pos = nx.spring_layout(G, k=2, seed=42)

    plt.figure(figsize=(14, 10))

    # Cyan for Drugs, Soft Red/Coral for Reactions
    palette = sns.color_palette("Set2")
    drug_color = palette[0]      #  Teal/Cyan
    reaction_color = palette[1]  #  Coral/Red

    # Separate nodes for coloring
    node_colors = []
    for node in G.nodes():
        if node in reactions_list:
            node_colors.append(reaction_color)
        else:
            node_colors.append(drug_color)

    # 1. Draw Edges with transparency
    weights = [G[u][v]['weight'] / 30 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='lightgray',
                           arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')

    # 2. Draw Nodes 
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors,
                           edgecolors='white', linewidths=2)

    # 3. Draw Labels 
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_family='sans-serif',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

    # 4. CREATE A CUSTOM LEGEND
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Drug / Combination',
               markerfacecolor=drug_color, markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Adverse Reaction',
               markerfacecolor=reaction_color, markersize=15),
        Line2D([0], [0], color='lightgray', lw=2, label='Association Strength (Lift)')
    ]

    plt.legend(handles=legend_elements, loc='upper right', title="Legend", frameon=True, fontsize=10)



    plt.suptitle("Pharmacovigilance Signal Discovery: Association Rule Network",
              fontsize=18, fontweight='bold', fontfamily='serif')
    plt.title(f"Algorithm: {algo_name} | Focus: {item_type} | {support_text}", 
              fontsize=14, color='gray', pad=25, fontfamily='sans-serif')

   
    sns.despine(left=True, bottom=True) # Remove the axis box
    plt.tight_layout()

    # 5. Save the plot with a clean name derived from the input CSV
    input_filename = os.path.basename(csv_path)
    clean_name = input_filename.replace("filtered_", "").replace(".csv", ".png")
    
    output_dir = "taskA/TaskAPlots"
    os.makedirs(output_dir, exist_ok=True)
    
    output_plot = os.path.join(output_dir, clean_name)
    plt.savefig(output_plot, dpi=300)
    print(f"Plot successfully saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Association Rule Network Plot")
    parser.add_argument("--file", required=True, help="Path to the filtered rules CSV file")
    
    args = parser.parse_args()
    generate_formal_plot(args.file)