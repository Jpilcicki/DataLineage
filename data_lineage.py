#Data Lineage Graph Generator
#This script searches for and .ipynb, .py, .csv, .json, and mentions of csv/json in read/write operations
#Then you get a nice, interactable graph to mess with
#Date modified: 10/16/24
#Author: Jacob Pilcicki
import os
import ast
import networkx as nx
import matplotlib.pyplot as plt
import nbformat
from textwrap import wrap
import math
import tkinter as tk
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
import json
import textwrap

def extract_io_operations(file_path):
    operations = []
    
    def extract_from_ast(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ['read_csv', 'read_json', 'to_csv', 'to_json']:
                        args = extract_args(node.args)
                        operations.append((func.attr, args))
                elif isinstance(func, ast.Name) and func.id == 'open':
                    args = extract_args(node.args)
                    operations.append(('open', args))
            elif isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and ('file_path' in node.targets[0].id or 'csv' in node.targets[0].id or 'json' in node.targets[0].id):
                    if isinstance(node.value, ast.Str):
                        operations.append(('file_assign', [node.value.s]))
                    elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr in ['replace', 'split']:
                            operations.append(('file_assign', [ast.unparse(node.value)]))

    def extract_args(args):
        extracted = []
        for arg in args:
            if isinstance(arg, ast.Str):
                extracted.append(arg.s)
            elif isinstance(arg, ast.Name):
                extracted.append(f"variable:{arg.id}")
            elif isinstance(arg, ast.Attribute):
                extracted.append(f"attribute:{ast.unparse(arg)}")
            elif isinstance(arg, ast.Call):
                extracted.append(f"call:{ast.unparse(arg)}")
            else:
                extracted.append(str(type(arg)))
        return extracted

    if file_path.endswith('.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
            extract_from_ast(tree)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    elif file_path.endswith('.ipynb'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                notebook = nbformat.read(file, as_version=4)
            
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    try:
                        tree = ast.parse(cell.source)
                        extract_from_ast(tree)
                    except SyntaxError:
                        continue
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return operations

def create_lineage_graph(directory):
    G = nx.DiGraph()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.ipynb')):
                file_path = os.path.join(root, file)
                operations = extract_io_operations(file_path)
                
                # Get two directories up
                dir_path = os.path.dirname(file_path)
                parent_dir = os.path.basename(dir_path)
                grandparent_dir = os.path.basename(os.path.dirname(dir_path))
                file_node = os.path.join(grandparent_dir, parent_dir, file)
                
                for op, args in operations:
                    if args:
                        file_name = args[0]
                        
                        if op == 'file_assign':
                            # Handle file path assignments
                            if "'" in file_name:
                                file_name = file_name.split("'")[-2]
                            elif '"' in file_name:
                                file_name = file_name.split('"')[-2]
                        
                        # Check if the file_name is a CSV or JSON (or similar)
                        if get_file_type(file_name) in ['csv', 'json', 'csv_like', 'json_like']:
                            if file_name.startswith(('variable:', 'attribute:', 'call:')):
                                # For variables, include the script's directory in the node name
                                file_name = f"{os.path.join(grandparent_dir, parent_dir)}:{file_name}"
                            else:
                                # For actual files, use the full path relative to the base directory
                                rel_path = os.path.relpath(os.path.join(root, file_name), directory)
                                file_name = os.path.join(*os.path.split(rel_path)[-2:])  # Get last two parts of the path
                            
                            if op.startswith('read') or op == 'open':
                                G.add_edge(file_name, file_node)
                            elif op.startswith('to'):
                                G.add_edge(file_node, file_name)
    
    return G


def create_spider_graph(directory):
    G = nx.DiGraph()
    file_paths = {}  # Dictionary to store normalized file paths

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.ipynb')):
                file_path = os.path.join(root, file)
                operations = extract_io_operations(file_path)
                
                file_node = os.path.relpath(file_path, directory)
                
                for op, args in operations:
                    if args:
                        file_name = args[0]
                        
                        if op == 'file_assign':
                            if "'" in file_name:
                                file_name = file_name.split("'")[-2]
                            elif '"' in file_name:
                                file_name = file_name.split('"')[-2]
                        
                        # Normalize the file path
                        if not os.path.isabs(file_name):
                            full_path = os.path.normpath(os.path.join(os.path.dirname(file_path), file_name))
                        else:
                            full_path = os.path.normpath(file_name)
                        
                        # Use the normalized path as the node name
                        normalized_path = os.path.relpath(full_path, directory)
                        
                        if normalized_path not in file_paths:
                            file_paths[normalized_path] = set()
                        file_paths[normalized_path].add(file_node)
                        
                        if op.startswith('read') or op == 'open':
                            G.add_edge(normalized_path, file_node)
                        elif op.startswith('to'):
                            G.add_edge(file_node, normalized_path)
    
    # Connect nodes that reference the same file
    for file_path, referencing_nodes in file_paths.items():
        for node1 in referencing_nodes:
            for node2 in referencing_nodes:
                if node1 != node2:
                    G.add_edge(node1, node2, color='blue', style='dashed')
    
    return G

def get_file_type(node):
    lower_node = node.lower()
    if node.endswith('.csv'):
        return 'csv'
    elif node.endswith('.json'):
        return 'json'
    elif node.endswith('.py'):
        return 'py'
    elif node.endswith('.ipynb'):
        return 'ipynb'
    elif 'csv' in lower_node:
        return 'csv_like'
    elif 'json' in lower_node:
        return 'json_like'
    else:
        return 'other'

def get_node_color(node):
    file_type = get_file_type(node)
    if file_type in ['csv', 'csv_like']:
        return '#FFA07A'  # Light Salmon for CSV
    elif file_type in ['json', 'json_like']:
        return '#98FB98'  # Pale Green for JSON
    elif file_type == 'py':
        return '#87CEFA'  # Light Sky Blue for PY
    elif file_type == 'ipynb':
        return '#DDA0DD'  # Plum for IPYNB
    else:
        return '#F0E68C'  # Khaki for Other

def get_screen_size():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height






def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/')

def trace_file_lineage(G, target_file):
    target_file = normalize_path(target_file)
    
    target_node = None
    for node in G.nodes():
        normalized_node = normalize_path(node)
        if (target_file in normalized_node or 
            normalized_node.endswith(target_file) or 
            os.path.basename(target_file) in normalized_node):
            target_node = node
            break
    
    if target_node is None:
        print(f"File {target_file} not found in the graph.")
        return None

    lineage_graph = nx.DiGraph()
    
    def add_predecessors(node):
        predecessors = list(G.predecessors(node))
        for pred in predecessors:
            lineage_graph.add_edge(pred, node)
            add_predecessors(pred)
    
    def add_successors(node):
        successors = list(G.successors(node))
        for succ in successors:
            lineage_graph.add_edge(node, succ)
            add_successors(succ)
    
    lineage_graph.add_node(target_node)
    add_predecessors(target_node)
    add_successors(target_node)
    
    return lineage_graph

def visualize_traced_lineage(G, lineage_graph, target_file, mode='interactive', context='sub'):
    if context == 'sub':
        viz_type = input("Choose visualization type for sub-graph (matplotlib/plotly): ").lower()
        if viz_type == 'matplotlib':
            visualize_graph_matplotlib(lineage_graph, mode, f"Lineage of {os.path.basename(target_file)}")
        elif viz_type == 'plotly':
            visualize_graph_plotly(lineage_graph, mode, f"Lineage of {os.path.basename(target_file)}")
        else:
            print("Invalid visualization type. Using matplotlib as default.")
            visualize_graph_matplotlib(lineage_graph, mode, f"Lineage of {os.path.basename(target_file)}")
    else:
        # Create a copy of the full graph
        full_graph = G.copy()
        
        # Highlight the nodes and edges in the lineage
        node_colors = []
        edge_colors = []
        
        for node in full_graph.nodes():
            if node in lineage_graph.nodes():
                node_colors.append('red')
            else:
                node_colors.append(get_node_color(node))
        
        for edge in full_graph.edges():
            if edge in lineage_graph.edges():
                edge_colors.append('red')
            else:
                edge_colors.append('lightgray')
        
        # Visualize the full graph with highlighted lineage
        plt.figure(figsize=(60, 45))
        pos = nx.spring_layout(full_graph, k=0.5, iterations=50)
        
        nx.draw_networkx_nodes(full_graph, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_edges(full_graph, pos, edge_color=edge_colors, width=1, arrows=True, arrowsize=10)
        
        labels = {node: '\n'.join(wrap(os.path.basename(node), 20)) for node in full_graph.nodes()}
        nx.draw_networkx_labels(full_graph, pos, labels, font_size=6)
        
        plt.title(f"Full Graph with Highlighted Lineage of {os.path.basename(target_file)}")
        plt.axis('off')
        plt.tight_layout()
        
        if mode == 'interactive':
            plt.show()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_file = os.path.join(os.path.dirname(__file__), f'full_graph_lineage_{os.path.basename(target_file)}_{timestamp}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Full graph with highlighted lineage saved as {output_file}")


def visualize_spider_graph(G, mode='interactive', title="Data Lineage Spider Diagram"):
    if not G.nodes():
        print("No data to visualize. The graph is empty.")
        return

    components = list(nx.weakly_connected_components(G))
    grid_size = math.ceil(math.sqrt(len(components)))
    
    fig_width = min(32 * (grid_size / 2), 60)  # Max width of 60 inches
    fig_height = min(24 * (grid_size / 2), 45)  # Max height of 45 inches
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    for i, component in enumerate(components):
        subgraph = G.subgraph(component)
        
        row = i // grid_size
        col = i % grid_size
        
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        pos = {node: (x + col * 4, -y - row * 4) for node, (x, y) in pos.items()}
        
        node_size = max(1000, 5000 / len(subgraph))
        
        # Draw nodes
        node_colors = [get_node_color(node) for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_size)
        
        # Draw edges
        normal_edges = [(u, v) for (u, v, d) in subgraph.edges(data=True) if 'style' not in d]
        dashed_edges = [(u, v) for (u, v, d) in subgraph.edges(data=True) if d.get('style') == 'dashed']
        
        nx.draw_networkx_edges(subgraph, pos, edgelist=normal_edges, edge_color='gray', 
                               arrows=True, arrowsize=20, node_size=node_size, arrowstyle='->')
        nx.draw_networkx_edges(subgraph, pos, edgelist=dashed_edges, edge_color='blue', 
                               style='dashed', arrows=False, node_size=node_size)
        
        # Draw labels
        font_size = max(3, 7 - len(subgraph) // 20)
        labels = {}
        for node in subgraph.nodes():
            parts = node.split(os.sep)
            if len(parts) >= 3:
                label = os.path.join(parts[-3], parts[-2], parts[-1])
            elif len(parts) == 2:
                label = os.path.join(parts[-2], parts[-1])
            else:
                label = parts[-1]
            labels[node] = '\n'.join(wrap(label, 20))
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=font_size, font_weight="bold")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='CSV', 
                                  markerfacecolor='#FFA07A', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='JSON', 
                                  markerfacecolor='#98FB98', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='PY', 
                                  markerfacecolor='#87CEFA', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='IPYNB', 
                                  markerfacecolor='#DDA0DD', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='Other', 
                                  markerfacecolor='#F0E68C', markersize=10),
                       plt.Line2D([0], [0], color='gray', label='Input/Output'),
                       plt.Line2D([0], [0], color='blue', linestyle='--', label='Same File Reference')]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplots_adjust(right=0.85)
    
    if mode == 'interactive':
        plt.show()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(os.path.dirname(__file__), f'{title.replace(" ", "_").lower()}_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Spider diagram saved as {output_file}")


def save_nodes_list(G):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = os.path.join(os.path.dirname(__file__), f"graph_nodes_{timestamp}.txt")
    with open(filename, 'w') as f:
        for node in sorted(G.nodes()):
            f.write(f"{node}\n")
    print(f"List of nodes saved to {filename}")


def visualize_graph_matplotlib(G, mode='interactive', title="Data Lineage Diagram"):
    if not G.nodes():
        print("No data to visualize. The graph is empty.")
        return

    components = list(nx.weakly_connected_components(G))
    grid_size = math.ceil(math.sqrt(len(components)))
    
    fig_width = min(32 * (grid_size / 2), 60)  # Max width of 60 inches
    fig_height = min(24 * (grid_size / 2), 45)  # Max height of 45 inches
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    for i, component in enumerate(components):
        subgraph = G.subgraph(component)
        
        row = i // grid_size
        col = i % grid_size
        
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        pos = {node: (x + col * 4, -y - row * 4) for node, (x, y) in pos.items()}
        
        node_size = max(1000, 5000 / len(subgraph))
        
        node_colors = [get_node_color(node) for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_size)
        
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=1, 
                               arrows=True, arrowsize=20, 
                               node_size=node_size, arrowstyle='->')
        
        font_size = (max(3, 7 - len(subgraph) // 20) -1)
        
        labels = {}
        for node in subgraph.nodes():
            parts = node.split(os.sep)
            if len(parts) >= 3:
                label = os.path.join(parts[-3], parts[-2], parts[-1])
            elif len(parts) == 2:
                label = os.path.join(parts[-2], parts[-1])
            else:
                label = parts[-1]
            labels[node] = '\n'.join(wrap(label, 20))
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=font_size, font_weight="bold")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='CSV', 
                                  markerfacecolor='#FFA07A', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='JSON', 
                                  markerfacecolor='#98FB98', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='PY', 
                                  markerfacecolor='#87CEFA', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='IPYNB', 
                                  markerfacecolor='#DDA0DD', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='Other', 
                                  markerfacecolor='#F0E68C', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplots_adjust(right=0.85)
    
    if mode == 'interactive':
        plt.show()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(os.path.dirname(__file__), f'{title.replace(" ", "_").lower()}_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Diagram saved as {output_file}")


def wrap_text(text, width=20):
    return '<br>'.join(textwrap.wrap(text, width=width))

def visualize_graph_plotly(G, mode='interactive', title="Data Lineage Diagram"):
    if not G.nodes():
        print("No data to visualize. The graph is empty.")
        return

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_labels = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(get_node_color(node))
        adjacencies = list(G.adj[node])
        node_text.append(f'{node}<br># of connections: {len(adjacencies)}')
        node_labels.append(wrap_text(node.split('\\')[-1], width=15))  # Wrap the filename

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,  # Use wrapped labels
        textposition="top center",
        textfont=dict(size=8),  # Smaller label size
        marker=dict(
            color=node_colors,
            size=15,  # Larger node size
            line_width=2),
        showlegend=False
    )

    # Create a list of traces for the legend
    legend_traces = []
    for file_type, color in [('CSV', '#FFA07A'), ('JSON', '#98FB98'), ('PY', '#87CEFA'), ('IPYNB', '#DDA0DD'), ('Other', '#F0E68C')]:
        legend_trace = go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=file_type, showlegend=True, name=file_type
        )
        legend_traces.append(legend_trace)

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
                    layout=go.Layout(
                        title=title,
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        legend=dict(
                            x=1.05,
                            y=1,
                            xanchor='left',
                            yanchor='top',
                            bgcolor='rgba(255, 255, 255, 0.5)',
                            bordercolor='rgba(0, 0, 0, 0.5)',
                            borderwidth=1
                        ),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Add arrows to the edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#888'
        )

    if mode == 'interactive':
        fig.show()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(os.path.dirname(__file__), f'{title.replace(" ", "_").lower()}_{timestamp}.html')
        pio.write_html(fig, file=output_file, auto_open=False)
        print(f"Interactive diagram saved as {output_file}")

if __name__ == "__main__":
    directory = input("Enter the directory path containing your Python scripts and Jupyter notebooks: ")
    G = create_lineage_graph(directory)
    
    while True:
        choice = input("Choose an option (full/trace/list/spider/quit): ").lower()
        if choice == 'full':
            viz_type = input("Choose visualization type (matplotlib/plotly): ").lower()
            mode = input("Choose visualization mode (interactive/png for matplotlib, interactive/html for plotly): ").lower()
            if viz_type == 'matplotlib':
                if mode in ['interactive', 'png']:
                    visualize_graph_matplotlib(G, mode)
                else:
                    print("Invalid mode for matplotlib. Please enter 'interactive' or 'png'.")
            elif viz_type == 'plotly':
                if mode in ['interactive', 'html']:
                    visualize_graph_plotly(G, mode)
                else:
                    print("Invalid mode for plotly. Please enter 'interactive' or 'html'.")
            else:
                print("Invalid visualization type. Please enter 'matplotlib' or 'plotly'.")
        elif choice == 'trace':
            target_file = input("Enter the full path or name of the file to trace: ")
            lineage_graph = trace_file_lineage(G, target_file)
            if lineage_graph:
                mode = input("Choose visualization mode (interactive/png): ").lower()
                context = input("Choose context (sub/full): ").lower()
                if mode in ['interactive', 'png'] and context in ['sub', 'full']:
                    visualize_traced_lineage(G, lineage_graph, target_file, mode, context)
                else:
                    print("Invalid input. Please enter 'interactive' or 'png' for mode, and 'sub' or 'full' for context.")
        elif choice == 'list':
            save_nodes_list(G)
        elif choice == 'spider':
            spider_G = create_spider_graph(directory)
            mode = input("Choose visualization mode (interactive/png): ").lower()
            if mode in ['interactive', 'png']:
                visualize_spider_graph(spider_G, mode)
            else:
                print("Invalid mode. Please enter 'interactive' or 'png'.")
        elif choice == 'quit':
            break
        else:
            print("Invalid choice. Please enter 'full', 'trace', 'list', 'spider', or 'quit'.")
