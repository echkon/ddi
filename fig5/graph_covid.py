import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

CSV_FILE = os.path.join(DATA_DIR, "ddi_covid.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "DDI_network_6_drugs_covid.png")


def plot_drug_network(df, fn=None, seed=50):

    # Standardize drug names
    df["drug_i"] = df["drug_i"].str.strip().str.upper()
    df["drug_j"] = df["drug_j"].str.strip().str.upper()

    # Build graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row["drug_i"],
            row["drug_j"],
            kind=row["kind"],
            weight=row["weight"],
        )

    pos = nx.spring_layout(G, seed=seed, k=0.5)

    # Separate edge types
    harmful_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "harm"]
    synergistic_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d["kind"] == "synergy"
    ]

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=4000,
        node_color="skyblue",
        alpha=0.9,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=harmful_edges,
        width=2.5,
        alpha=0.8,
        edge_color="red",
        style="solid",
        label="Harmful Interaction",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=synergistic_edges,
        width=2.5,
        alpha=0.8,
        edge_color="green",
        style="dashed",
        label="Synergistic Interaction",
    )

    nx.draw_networkx_labels(G, pos, font_size=26)

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="black",
        font_size=20,
    )

    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="red", lw=4, label="Harmful"),
            plt.Line2D([0], [0], color="green", lw=4, ls="--", label="Synergistic"),
        ],
        loc="best",
        fontsize=24,
    )

    plt.axis("off")
    plt.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        print(f"Graph saved to file: {fn}")

    else:
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    plot_drug_network(df, fn=OUTPUT_FILE, seed=45)
