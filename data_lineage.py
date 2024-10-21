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
        visualize_graph(lineage_graph, mode, f"Lineage of {os.path.basename(target_file)}")
    else:
        # Create a copy of the full graph
        full_graph = G.copy()
        
        # Identify connected components (subgraphs)
        components = list(nx.weakly_connected_components(full_graph))
        
        # Calculate the complexity of each subgraph
        complexities = [len(component) for component in components]
        total_complexity = sum(complexities)
        
        # Calculate the figure size based on the total complexity
        base_size = 20
        fig_size = max(base_size, base_size * math.log(total_complexity))
        plt.figure(figsize=(fig_size, fig_size))
        
        # Custom positioning for subgraphs
        def position_subgraphs(components, complexities):
            positions = {}
            grid_size = math.ceil(math.sqrt(len(components)))
            total_area = grid_size ** 2
            scale = math.sqrt(total_area / total_complexity)
            
            x, y = 0, 0
            for component, complexity in zip(components, complexities):
                subgraph = full_graph.subgraph(component)
                size = math.sqrt(complexity) * scale
                sub_pos = nx.spring_layout(subgraph)
                
                # Scale and translate the subgraph
                for node, pos in sub_pos.items():
                    positions[node] = ((pos[0] * size + x), (pos[1] * size + y))
                
                x += size
                if x > grid_size:
                    x = 0
                    y += size
            
            return positions
        
        pos = position_subgraphs(components, complexities)
        
        # Highlight the nodes and edges in the lineage
        node_colors = []
        edge_colors = []
        node_sizes = []
        
        for node in full_graph.nodes():
            if node in lineage_graph.nodes():
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append(get_node_color(node))
                node_sizes.append(100)
        
        for edge in full_graph.edges():
            if edge in lineage_graph.edges():
                edge_colors.append('red')
            else:
                edge_colors.append('lightgray')
        
        # Draw the graph
        nx.draw_networkx_nodes(full_graph, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(full_graph, pos, edge_color=edge_colors, width=0.5, arrows=True, arrowsize=5)
        
        # Add labels only for the nodes in the lineage
        labels = {node: '\n'.join(wrap(os.path.basename(node), 20)) for node in lineage_graph.nodes()}
        nx.draw_networkx_labels(full_graph, pos, labels, font_size=6)
        
        plt.title(f"Full Graph with Highlighted Lineage of {os.path.basename(target_file)}")
        plt.axis('off')
        plt.tight_layout()
        
        if mode == 'interactive':
            plt.show()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_file = f'full_graph_lineage_{os.path.basename(target_file)}_{timestamp}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Full graph with highlighted lineage saved as {output_file}")


def save_nodes_list(G):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"graph_nodes_{timestamp}.txt"
    with open(filename, 'w') as f:
        for node in sorted(G.nodes()):
            f.write(f"{node}\n")
    print(f"List of nodes saved to {filename}")


def visualize_graph(G, mode='interactive', title="Data Lineage Diagram"):
    if not G.nodes():
        print("No data to visualize. The graph is empty.")
        return

    components = list(nx.weakly_connected_components(G))
    grid_size = math.ceil(math.sqrt(len(components)))
    
    fig_width = min(32 * (grid_size / 2), 20)  # Max width of 20 inches
    fig_height = min(24 * (grid_size / 2), 15)  # Max height of 15 inches
    
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
                                  markerfacecolor='#F0E68C', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplots_adjust(right=0.85)
    
    if mode == 'interactive':
        plt.show()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = f'{title.replace(" ", "_").lower()}_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Diagram saved as {output_file}")

if __name__ == "__main__":
    directory = input("Enter the directory path containing your Python scripts and Jupyter notebooks: ")
    G = create_lineage_graph(directory)
    
    while True:
        choice = input("Choose an option (full/trace/list/quit): ").lower()
        if choice == 'full':
            mode = input("Choose visualization mode (interactive/png): ").lower()
            if mode in ['interactive', 'png']:
                visualize_graph(G, mode)
            else:
                print("Invalid mode. Please enter 'interactive' or 'png'.")
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
        elif choice == 'quit':
            break
        else:
            print("Invalid choice. Please enter 'full', 'trace', 'list', or 'quit'.")
