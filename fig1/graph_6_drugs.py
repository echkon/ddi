import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

df = pd.read_csv(os.path.join(DATA_DIR, "ddi_r.csv"))


def plot_drug_network(df, fn=None):
    # Standardize drug names
    df["drug_i"] = df["drug_i"].str.strip().str.title()
    df["drug_j"] = df["drug_j"].str.strip().str.title()

    # Initialize graph
    G = nx.Graph()

    # Add edges with interaction type and weight
    for _, row in df.iterrows():
        G.add_edge(
            row["drug_i"],
            row["drug_j"],
            kind=row["kind"],
            weight=row["weight"],
        )

    # Graph layout
    pos = {
        "Everolimus": (0.0, 1.0),
        "Topotecan": (-1.2, 0.6),
        "Metformin": (1.2, 0.6),
        "Cabazitaxel": (-0.6, -1.0),
        "Erlotinib": (0.8, -1.0),
        "Ritonavir": (0.2, -0.1),
    }

    pos = pos

    # Separate edge types
    harmful_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "harm"]
    synergistic_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d["kind"] == "synergy"
    ]

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=800,
        node_color="skyblue",
    )

    # Draw harmful interactions
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=harmful_edges,
        width=2.0,
        alpha=0.8,
        edge_color="red",
        style="solid",
        label="Harmful Interaction",
    )

    # Draw synergistic interactions
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=synergistic_edges,
        width=2.0,
        alpha=0.8,
        edge_color="green",
        style="dashed",
        label="Synergistic Interaction",
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="black",
    )

    plt.title("Drug–Drug Interaction Network", fontsize=16)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="red", lw=2, label="harmful"),
            plt.Line2D(
                [0],
                [0],
                color="green",
                lw=2,
                ls="--",
                label="synergy",
            ),
        ],
        loc="best",
    )

    plt.axis("off")

    if fn:
        plt.savefig(fn, bbox_inches="tight")
        print(f"Graph saved to file: {fn}")
    else:
        plt.show()


if __name__ == "__main__":
    output_file = os.path.join(BASE_DIR, "DDI_network_6_drugs.png")
    plot_drug_network(df, fn=output_file)
