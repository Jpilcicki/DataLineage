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

def visualize_graph(G):
    if not G.nodes():
        print("No data to visualize. The graph is empty.")
        return

    components = list(nx.weakly_connected_components(G))
    grid_size = math.ceil(math.sqrt(len(components)))
    
    # Get screen size
    screen_width, screen_height = get_screen_size()
    
    # Calculate figure size based on screen size
    fig_width = min(32 * (grid_size / 2), screen_width / 100)  # Convert pixels to inches
    fig_height = min(24 * (grid_size / 2), screen_height / 100)  # Convert pixels to inches
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
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

    plt.title("Data Lineage Diagram")
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
    
    # Maximize the window (this should work for most backends)
    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'window'):
        if hasattr(mng.window, 'state'):
            mng.window.state('zoomed')  # For TkAgg on Windows
        elif hasattr(mng.window, 'showMaximized'):
            mng.window.showMaximized()  # For Qt backend
        elif hasattr(mng.window, 'maximize'):
            mng.window.maximize()  # For wxPython
    elif hasattr(mng, 'frame'):
        mng.frame.Maximize(True)  # For WX backend
    elif hasattr(mng, 'full_screen_toggle'):
        mng.full_screen_toggle()  # For Qt5Agg backend on some systems
    elif hasattr(mng, 'resize'):
        mng.resize(*mng.window.maxsize())  # For TkAgg on non-Windows systems
    
    plt.show()

if __name__ == "__main__":
    directory = input("Enter the directory path containing your Python scripts and Jupyter notebooks: ")
    G = create_lineage_graph(directory)
    visualize_graph(G)