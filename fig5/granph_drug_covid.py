import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io


def plot_drug_network(df, fn=None, seed=50):

    df["drug_i"] = df["drug_i"].str.strip().str.upper()
    df["drug_j"] = df["drug_j"].str.strip().str.upper()

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["drug_i"], row["drug_j"], kind=row["kind"], weight=row["weight"])

    # --- SETUP LAYOUT ---
    pos = nx.spring_layout(G, seed=seed, k=0.5)

    harm_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "harm"]
    synergy_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "synergy"]
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # --- VẼ ĐỒ THỊ ---
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color="skyblue", alpha=0.9)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=harm_edges,
        width=2.5,
        alpha=0.8,
        edge_color="red",
        style="solid",
        label="Harmful Interaction",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=synergy_edges,
        width=2.5,
        alpha=0.8,
        edge_color="green",
        style="dashed",
        label="Synergistic Interaction",
    )

    nx.draw_networkx_labels(G, pos, font_size=26)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="black", font_size=20
    )

    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="red", lw=4, label="harmful"),
            plt.Line2D([0], [0], color="green", lw=4, ls="--", label="synergy"),
        ],
        loc="best",
        fontsize=26,
    )
    plt.axis("off")
    plt.tight_layout()

    if fn:

        plt.savefig(fn, bbox_inches="tight", dpi=300)
        print(f"Đã lưu đồ thị vào file: {fn}")

        eps_fn = fn.rsplit(".", 1)[0] + ".eps" if "." in fn else fn + ".eps"

        plt.savefig(eps_fn, format="eps", bbox_inches="tight")
        print(f"Đã lưu đồ thị vào file EPS: {eps_fn}")
    else:
        plt.show()


try:
    drug_df = pd.read_csv("ddi_covid.csv")
    plot_drug_network(drug_df, fn="Drug_graph_covid.png", seed=45)
except FileNotFoundError:
    print("Không tìm thấy file csv.")
