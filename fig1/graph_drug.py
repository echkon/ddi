import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io


def plot_drug_network(df, fn=None):

    df["drug_i"] = df["drug_i"].str.strip().str.title()
    df["drug_j"] = df["drug_j"].str.strip().str.title()

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["drug_i"], row["drug_j"], kind=row["kind"], weight=row["weight"])

    pos = nx.kamada_kawai_layout(G, seed=10)

    harm_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "harm"]
    synergy_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "synergy"]

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue")

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=harm_edges,
        width=2.0,
        alpha=0.8,
        edge_color="red",
        style="solid",
        label="Harmful Interaction",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=synergy_edges,
        width=2.0,
        alpha=0.8,
        edge_color="green",
        style="dashed",
        label="Synergistic Interaction",
    )

    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black")

    plt.title("Drug-Drug Interaction Network", fontsize=16)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="red", lw=2, label="Harmful interaction"),
            plt.Line2D(
                [0],
                [0],
                color="green",
                lw=2,
                ls="--",
                label="Synergistic interaction",
            ),
        ],
        loc="best",
    )
    plt.axis("off")

    if fn:
        plt.savefig(fn, bbox_inches="tight")
        print(f"Đã lưu đồ thị vào file: {fn}")
    else:
        plt.show()


drug_df = pd.read_csv("ddi_r.csv")

plot_drug_network(drug_df, fn="Drug_graph_paper")
