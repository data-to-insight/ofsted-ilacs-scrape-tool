import yaml
import matplotlib.pyplot as plt
import networkx as nx
import warnings

from networkx.drawing.nx_pydot import graphviz_layout

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load YAML
yaml_path = "./sccm.yml"
with open(yaml_path, "r") as file:
    sccm_data = yaml.safe_load(file)

# Extract project name
project_name = sccm_data.get("project", {}).get("name", "Unnamed Project")
relationships = sccm_data.get("sccm_alignment", {}).get("relationships", [])

# Collect entities that are used in relationships (subjects or objects)
used_entities = set()
for rel in relationships:
    used_entities.update([rel.get("subject"), rel.get("object")])

# Filter only the entities used in relationships
entities = [
    e.get("name") for e in sccm_data.get("sccm_alignment", {}).get("entities", [])
    if e.get("name") in used_entities
]

# Build a label mapping from name → label
label_lookup = {}
for e in sccm_data.get("sccm_alignment", {}).get("entities", []):
    name = e.get("name")
    label = e.get("label", name)
    if name in used_entities:
        label_lookup[name] = label
label_lookup[project_name] = project_name

# Create the graph
G = nx.DiGraph()
G.add_node(project_name, shape='box', color='lightblue')

# Add relevant entities as nodes
for entity in entities:
    G.add_node(entity)

# Tool-specific semantic links
tool_relationships = [
    (project_name, "InspectionSummary", "produces"),
    (project_name, "Ofsted", "analyses")
]

for src, dst, label in tool_relationships:
    if dst in G.nodes:
        G.add_edge(src, dst, label=label)

# Add all semantic relationships
for rel in relationships:
    subj = rel.get("subject")
    obj = rel.get("object")
    pred = rel.get("predicate")
    if subj and obj and pred:
        G.add_edge(subj, obj, label=pred)

# Generate edge labels
edge_labels = {(src, dst): data['label'] for src, dst, data in G.edges(data=True)}

# Layout and drawing
pos = nx.spring_layout(G, k=3.5, seed=42)
nx.draw(G, pos,
        labels=label_lookup,
        node_size=2500,
        node_color="lightblue",
        font_size=9,
        font_weight="bold",
        edge_color="grey",
        arrows=True)

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.65,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
)

plt.title("SCCM Graph – " + project_name, fontsize=14)
output_path = "./sccm_graph_static.png"
plt.savefig(output_path, dpi=300)
