import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = {
    "Low": "#52d3b1",
    "Moderate": "#f9c76d",
    "High": "#fa9c62",
    "Very High": "#f17191",
}
PALETTE = ["#52e0db", "#a78bfa", "#f9c76d", "#f17191", "#79a9f5", "#52d3b1"]


def styled(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#b6c6dc"),
        height=height,
        margin=dict(l=15, r=15, t=35, b=25),
        colorway=PALETTE,
        legend=dict(orientation="h", y=1.15),
        transition_duration=500,
    )
    fig.update_xaxes(gridcolor="#263247")
    fig.update_yaxes(gridcolor="#263247")
    return fig


def gauge(score, band):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 30, "color": COLORS[band]}},
            title={"text": band.upper() + " SUSPICION", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": COLORS[band], "thickness": 0.35},
                "bgcolor": "#1b2840",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#183d37"},
                    {"range": [30, 60], "color": "#3e3621"},
                    {"range": [60, 80], "color": "#483025"},
                    {"range": [80, 100], "color": "#472739"},
                ],
            },
        )
    )
    return styled(fig, 270)


def contributions(result):
    from reviewguard.scoring import NAMES

    return styled(
        go.Figure(
            go.Waterfall(
                x=[NAMES[k] for k in result.contributions] + ["Total"],
                y=list(result.contributions.values()) + [0],
                measure=["relative"] * 5 + ["total"],
                increasing={"marker": {"color": "#a78bfa"}},
                totals={"marker": {"color": "#52e0db"}},
                text=[f"{v:.1f}" for v in result.contributions.values()]
                + [f"{result.score:.1f}"],
                textposition="outside",
            )
        ),
        330,
    )


def network(result, threshold=0.85):
    import networkx as nx

    count = min(800, len(result.reviews))
    G = nx.Graph()
    G.add_nodes_from(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            if result.similarity[i, j] >= threshold:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42, iterations=40)
    x = []
    y = []
    for a, b in G.edges():
        x.extend([pos[a][0], pos[b][0], None])
        y.extend([pos[a][1], pos[b][1], None])
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=0.8, color="#536280"),
            hoverinfo="skip",
        )
    )
    labels = list(nx.connected_components(G))
    cluster = {i: k for k, g in enumerate(labels) for i in g}
    texts = [
        f"Review {i+1} · {result.reviews.suspicion_score.iloc[i]:.1f}/100<br>"
        + str(result.reviews.text.iloc[i])[:180]
        for i in range(count)
    ]
    fig.add_trace(
        go.Scatter(
            x=[pos[i][0] for i in range(count)],
            y=[pos[i][1] for i in range(count)],
            mode="markers",
            marker=dict(
                size=[
                    8 + result.reviews.suspicion_score.iloc[i] / 8 for i in range(count)
                ],
                color=[
                    PALETTE[cluster[i] % len(PALETTE)] if G.degree(i) else "#34445b"
                    for i in range(count)
                ],
                line=dict(width=1, color="#111a29"),
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig = styled(fig, 520)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
