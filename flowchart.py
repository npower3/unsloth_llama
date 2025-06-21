import openai
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import re
from typing import Dict, List, Any, Tuple
import networkx as nx
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import webbrowser
import os
import tempfile

class DynamicFlowchartGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize the Dynamic Flowchart Generator
        
        Args:
            api_key (str): OpenAI API key for GPT-3
        """
        if api_key:
            openai.api_key = api_key
        else:
            print("⚠️  No API key provided. Please set it using set_api_key() method")
        
        self.flowchart_data = None
        self.fig = None
        self.ax = None
        self.graph = nx.DiGraph()
        
    def set_api_key(self, api_key: str):
        """Set OpenAI API key"""
        openai.api_key = api_key
        
    def convert_text_to_flowchart_data(self, text: str, process_name: str = "Business Process") -> Dict[str, Any]:
        """
        Use GPT-3 to convert free text business rules to structured flowchart data
        
        Args:
            text (str): Business rules text
            process_name (str): Name of the process
            
        Returns:
            Dict: Structured flowchart data
        """
        
        prompt = f"""
        You are an expert business analyst. Convert the following business rules text into a structured JSON format for creating a flowchart.

        IMPORTANT: Create a complete flowchart structure with the following exact format:

        {{
            "process_name": "{process_name}",
            "nodes": [
                {{
                    "id": "unique_node_id",
                    "type": "start|decision|process|end",
                    "label": "Short node label",
                    "description": "Detailed description",
                    "position": [x, y]
                }}
            ],
            "connections": [
                {{
                    "from": "source_node_id",
                    "to": "target_node_id", 
                    "label": "condition or action",
                    "type": "yes|no|default"
                }}
            ]
        }}

        Rules for conversion:
        1. Always start with a "start" node
        2. Create "decision" nodes for any conditions, checks, validations, or if/then statements
        3. Create "process" nodes for actions, assignments, calculations, or operations
        4. Always end with "end" nodes for different outcomes
        5. Extract ALL specific details like field names, values, conditions
        6. Create separate paths for different outcomes (success, error, bypass, etc.)
        7. Include exact filter conditions, status codes, and field assignments
        8. Position nodes logically from top to bottom, left to right

        Business Rules Text:
        {text}

        Return ONLY the JSON structure, no additional text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 which is more cost-effective
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at converting business rules to flowchart structures. Always return valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response to extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # Parse JSON
            flowchart_data = json.loads(response_text)
            
            # Validate and fix structure if needed
            flowchart_data = self._validate_and_fix_structure(flowchart_data)
            
            return flowchart_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response_text}")
            return self._create_fallback_structure(text, process_name)
            
        except Exception as e:
            print(f"GPT-3 API error: {e}")
            return self._create_fallback_structure(text, process_name)
    
    def _validate_and_fix_structure(self, data: Dict) -> Dict:
        """Validate and fix flowchart structure"""
        
        # Ensure required keys exist
        if "nodes" not in data:
            data["nodes"] = []
        if "connections" not in data:
            data["connections"] = []
        if "process_name" not in data:
            data["process_name"] = "Business Process"
        
        # Auto-assign positions if missing
        for i, node in enumerate(data["nodes"]):
            if "position" not in node or not node["position"]:
                node["position"] = [5, 10 - i * 1.5]
            
            # Ensure required node fields
            if "id" not in node:
                node["id"] = f"node_{i}"
            if "type" not in node:
                node["type"] = "process"
            if "label" not in node:
                node["label"] = f"Step {i+1}"
            if "description" not in node:
                node["description"] = node.get("label", "")
        
        return data
    
    def _create_fallback_structure(self, text: str, process_name: str) -> Dict:
        """Create a basic flowchart structure when GPT-3 fails"""
        
        # Simple text analysis for fallback
        sentences = text.split('.')
        
        nodes = [
            {
                "id": "start",
                "type": "start", 
                "label": "Start Process",
                "description": f"Begin {process_name}",
                "position": [5, 10]
            }
        ]
        
        connections = []
        current_y = 8.5
        prev_node = "start"
        
        for i, sentence in enumerate(sentences[:5]):  # Limit to 5 sentences
            if sentence.strip():
                node_id = f"step_{i}"
                
                # Simple rule classification
                if any(word in sentence.lower() for word in ['if', 'check', 'validate', 'when', '?']):
                    node_type = "decision"
                else:
                    node_type = "process"
                
                nodes.append({
                    "id": node_id,
                    "type": node_type,
                    "label": sentence.strip()[:50] + "..." if len(sentence.strip()) > 50 else sentence.strip(),
                    "description": sentence.strip(),
                    "position": [5, current_y]
                })
                
                connections.append({
                    "from": prev_node,
                    "to": node_id,
                    "label": "",
                    "type": "default"
                })
                
                prev_node = node_id
                current_y -= 1.5
        
        # Add end node
        nodes.append({
            "id": "end",
            "type": "end",
            "label": "End Process", 
            "description": f"Complete {process_name}",
            "position": [5, current_y]
        })
        
        connections.append({
            "from": prev_node,
            "to": "end",
            "label": "",
            "type": "default"
        })
        
        return {
            "process_name": process_name,
            "nodes": nodes,
            "connections": connections
        }
    
    def create_flowchart(self, flowchart_data: Dict, interactive: bool = True):
        """
        Create visual flowchart from structured data
        
        Args:
            flowchart_data (Dict): Structured flowchart data
            interactive (bool): Whether to make it interactive
        """
        
        self.flowchart_data = flowchart_data
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 12))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 12)
        self.ax.axis('off')
        
        # Colors for different node types
        colors = {
            'start': '#4CAF50',      # Green
            'end': '#F44336',        # Red
            'decision': '#2196F3',   # Blue  
            'process': '#FF9800',    # Orange
            'filter': '#9C27B0'      # Purple
        }
        
        # Store node positions
        node_positions = {}
        
        # Draw nodes
        for node in flowchart_data['nodes']:
            x, y = node['position']
            node_positions[node['id']] = (x, y)
            
            # Choose color
            color = colors.get(node['type'], '#CCCCCC')
            
            # Draw node based on type
            if node['type'] == 'decision':
                # Diamond for decisions
                diamond = mpatches.RegularPolygon(
                    (x, y), 4, radius=0.8, 
                    orientation=np.pi/4,
                    facecolor=color, 
                    edgecolor='black', 
                    linewidth=2,
                    alpha=0.8
                )
                self.ax.add_patch(diamond)
                
            elif node['type'] in ['start', 'end']:
                # Oval for start/end
                ellipse = mpatches.Ellipse(
                    (x, y), 1.6, 0.8,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.8
                )
                self.ax.add_patch(ellipse)
                
            else:
                # Rectangle for process
                rect = FancyBboxPatch(
                    (x-0.8, y-0.4), 1.6, 0.8,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black', 
                    linewidth=2,
                    alpha=0.8
                )
                self.ax.add_patch(rect)
            
            # Add text
            self.ax.text(
                x, y, node['label'], 
                ha='center', va='center',
                fontsize=9, weight='bold',
                wrap=True, color='white'
            )
            
            # Add hover functionality if interactive
            if interactive:
                self._add_hover_effect(x, y, node)
        
        # Draw connections
        for conn in flowchart_data['connections']:
            if conn['from'] in node_positions and conn['to'] in node_positions:
                x1, y1 = node_positions[conn['from']]
                x2, y2 = node_positions[conn['to']]
                
                # Draw arrow
                self.ax.annotate(
                    '', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', 
                        lw=2, 
                        color='black',
                        connectionstyle="arc3,rad=0.1"
                    )
                )
                
                # Add label if present
                if conn.get('label'):
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    label_color = 'green' if conn.get('type') == 'yes' else 'red' if conn.get('type') == 'no' else 'black'
                    
                    self.ax.text(
                        mid_x + 0.3, mid_y, conn['label'],
                        fontsize=8, weight='bold',
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor='white',
                            edgecolor=label_color,
                            alpha=0.9
                        ),
                        color=label_color
                    )
        
        # Add title
        self.ax.text(
            5, 11.5, flowchart_data.get('process_name', 'Business Process Flowchart'),
            ha='center', va='center', 
            fontsize=16, weight='bold'
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=colors['start'], label='Start/End'),
            mpatches.Patch(color=colors['decision'], label='Decision'),
            mpatches.Patch(color=colors['process'], label='Process')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if interactive:
            # Add interactive buttons
            self._add_interactive_buttons()
        
        plt.show()
    
    def _add_hover_effect(self, x: float, y: float, node: Dict):
        """Add hover effect to show node details"""
        
        def on_hover(event):
            if event.inaxes == self.ax:
                # Check if mouse is over this node
                if abs(event.xdata - x) < 0.8 and abs(event.ydata - y) < 0.4:
                    # Show tooltip
                    tooltip_text = f"{node['label']}\n\n{node.get('description', '')}"
                    self.ax.text(
                        x, y-1, tooltip_text,
                        ha='center', va='top',
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor='yellow',
                            alpha=0.9
                        )
                    )
                    self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def _add_interactive_buttons(self):
        """Add interactive buttons to the plot"""
        
        # Export button
        ax_export = plt.axes([0.02, 0.02, 0.1, 0.04])
        button_export = Button(ax_export, 'Export')
        button_export.on_clicked(self._export_flowchart)
        
        # Regenerate button  
        ax_regen = plt.axes([0.13, 0.02, 0.1, 0.04])
        button_regen = Button(ax_regen, 'Regenerate')
        button_regen.on_clicked(self._regenerate_layout)
    
    def _export_flowchart(self, event):
        """Export flowchart to file"""
        if self.fig:
            filename = f"flowchart_{self.flowchart_data.get('process_name', 'process').replace(' ', '_')}.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Flowchart exported as {filename}")
    
    def _regenerate_layout(self, event):
        """Regenerate flowchart layout"""
        if self.flowchart_data:
            # Auto-arrange nodes using networkx
            self._auto_arrange_nodes()
            self.ax.clear()
            self.create_flowchart(self.flowchart_data, interactive=True)
    
    def _auto_arrange_nodes(self):
        """Automatically arrange nodes using graph layout"""
        
        # Build networkx graph
        G = nx.DiGraph()
        for node in self.flowchart_data['nodes']:
            G.add_node(node['id'])
        
        for conn in self.flowchart_data['connections']:
            G.add_edge(conn['from'], conn['to'])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Update node positions
        for node in self.flowchart_data['nodes']:
            if node['id'] in pos:
                x, y = pos[node['id']]
                # Scale and translate to fit our coordinate system
                node['position'] = [x * 4 + 5, y * 6 + 6]
    
    def create_gui_interface(self):
        """Create a GUI interface for easy use"""
        
        root = tk.Tk()
        root.title("Dynamic Flowchart Generator")
        root.geometry("800x600")
        
        # API Key input
        api_frame = ttk.Frame(root, padding="10")
        api_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(api_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W)
        self.api_key_var = tk.StringVar()
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        api_entry.grid(row=0, column=1, padx=(10, 0))
        
        ttk.Button(api_frame, text="Set API Key", 
                  command=self._set_api_key_from_gui).grid(row=0, column=2, padx=(10, 0))
        
        # Process name input
        name_frame = ttk.Frame(root, padding="10")
        name_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(name_frame, text="Process Name:").grid(row=0, column=0, sticky=tk.W)
        self.process_name_var = tk.StringVar(value="Business Process")
        ttk.Entry(name_frame, textvariable=self.process_name_var, width=50).grid(row=0, column=1, padx=(10, 0))
        
        # Text input area
        text_frame = ttk.Frame(root, padding="10")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(text_frame, text="Business Rules Text:").pack(anchor=tk.W)
        
        self.text_input = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            width=90, 
            height=20,
            font=("Arial", 10)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Default example text
        example_text = """The transformation logic for grgr_sts is as follows:
        
1. Apply Group ID Filter: Check if HP Group ID matches any value in t_cbms_attribute where attribute_name='GROUP ID FILTER'. If matched, set transaction_status='256' and record_status='215', then bypass further processing.

2. Apply Benefit ID Filter: Check if HP Benefit ID matches any value in t_cbms_attribute where attribute_name='BENEFIT ID FILTER'. If matched, set transaction_status='256' and record_status='215', then bypass further processing.

3. Validate Dates: Check if Employee Group Effective Date and Term Date are present and valid. If missing or invalid, set transaction_status='256' and record_status='216', then bypass further processing.

4. Assign Field: If all validations pass, assign grgr_sts based on grgr_term_dt using GroupTermDate logic.

5. Final Check: Verify all processing completed successfully. If any errors occurred, set grgr_sts to NULL."""
        
        self.text_input.insert(tk.END, example_text)
        
        # Buttons frame
        button_frame = ttk.Frame(root, padding="10")
        button_frame.grid(row=3, column=0)
        
        ttk.Button(button_frame, text="Generate Flowchart", 
                  command=self._generate_from_gui).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Text", 
                  command=self._clear_text).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Load Example", 
                  command=self._load_example).pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready. Enter your OpenAI API key and business rules text.")
        status_label = ttk.Label(root, textvariable=self.status_var)
        status_label.grid(row=4, column=0, pady=(5, 0))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        root.mainloop()
    
    def _set_api_key_from_gui(self):
        """Set API key from GUI input"""
        api_key = self.api_key_var.get().strip()
        if api_key:
            self.set_api_key(api_key)
            self.status_var.set("API key set successfully!")
        else:
            messagebox.showerror("Error", "Please enter a valid API key")
    
    def _generate_from_gui(self):
        """Generate flowchart from GUI input"""
        
        def generate_async():
            try:
                self.status_var.set("Generating flowchart... Please wait.")
                
                text = self.text_input.get(1.0, tk.END).strip()
                process_name = self.process_name_var.get().strip()
                
                if not text:
                    messagebox.showerror("Error", "Please enter business rules text")
                    return
                
                if not openai.api_key:
                    messagebox.showerror("Error", "Please set your OpenAI API key first")
                    return
                
                # Generate flowchart data
                flowchart_data = self.convert_text_to_flowchart_data(text, process_name)
                
                # Create flowchart
                self.create_flowchart(flowchart_data, interactive=True)
                
                self.status_var.set("Flowchart generated successfully!")
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to generate flowchart: {str(e)}")
        
        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=generate_async)
        thread.daemon = True
        thread.start()
    
    def _clear_text(self):
        """Clear text input"""
        self.text_input.delete(1.0, tk.END)
        self.status_var.set("Text cleared. Enter new business rules.")
    
    def _load_example(self):
        """Load example text"""
        example = """Process customer order validation:

1. Check if customer exists: Validate customer ID against customer database. If not found, create error record with status 'INVALID_CUSTOMER' and stop processing.

2. Verify payment method: Check if payment method is valid and has sufficient funds. If payment fails, set order status to 'PAYMENT_FAILED' and send notification to customer.

3. Check inventory: Verify product availability in inventory system. If insufficient stock, set status to 'OUT_OF_STOCK' and create backorder.

4. Calculate shipping: Determine shipping cost based on customer location and product weight. If shipping address is invalid, mark as 'INVALID_ADDRESS'.

5. Process order: If all validations pass, create order record with status 'CONFIRMED' and initiate fulfillment process.

6. Send confirmation: Email order confirmation to customer with tracking information."""
        
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, example)
        self.status_var.set("Example loaded. Click 'Generate Flowchart' to create visualization.")


# Example usage functions
def quick_generate(text: str, api_key: str, process_name: str = "Business Process"):
    """Quick function to generate flowchart from text"""
    
    generator = DynamicFlowchartGenerator(api_key)
    flowchart_data = generator.convert_text_to_flowchart_data(text, process_name)
    generator.create_flowchart(flowchart_data, interactive=True)
    return generator

def launch_gui():
    """Launch the GUI interface"""
    generator = DynamicFlowchartGenerator()
    generator.create_gui_interface()

# Example usage
if __name__ == "__main__":
    print("Dynamic Flowchart Generator")
    print("=" * 50)
    print("Options:")
    print("1. Launch GUI interface: launch_gui()")
    print("2. Quick generate: quick_generate(text, api_key, process_name)")
    print()
    print("Launching GUI...")
    launch_gui()
