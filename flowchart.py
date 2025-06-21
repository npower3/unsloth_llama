import requests

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import graphviz
from graphviz import Digraph
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import tempfile
import os
import webbrowser

class SophisticatedFlowchartGenerator:
    def __init__(self, custom_gpt4_endpoint: str):
        """
        Sophisticated flowchart generator using advanced visualization packages
        
        Args:
            custom_gpt4_endpoint (str): Your custom GPT-4 API endpoint
        """
        self.gpt4_endpoint = custom_gpt4_endpoint
        self.flowchart_data = None
        
        # Professional color schemes
        self.color_schemes = {
            'modern': {
                'start': {'bg': '#2E8B57', 'border': '#1F5F3F', 'text': '#FFFFFF'},
                'end': {'bg': '#DC143C', 'border': '#8B0000', 'text': '#FFFFFF'},
                'decision': {'bg': '#4169E1', 'border': '#191970', 'text': '#FFFFFF'},
                'process': {'bg': '#FF8C00', 'border': '#FF4500', 'text': '#FFFFFF'},
                'filter': {'bg': '#9932CC', 'border': '#4B0082', 'text': '#FFFFFF'}
            },
            'corporate': {
                'start': {'bg': '#0066CC', 'border': '#004499', 'text': '#FFFFFF'},
                'end': {'bg': '#CC0000', 'border': '#990000', 'text': '#FFFFFF'},
                'decision': {'bg': '#6699FF', 'border': '#3366CC', 'text': '#FFFFFF'},
                'process': {'bg': '#66CC99', 'border': '#339966', 'text': '#FFFFFF'},
                'filter': {'bg': '#FF9966', 'border': '#CC6633', 'text': '#FFFFFF'}
            }
        }
    
    def text_to_flowchart(self, business_rules_text: str, process_name: str = "Business Process", 
                         output_format: str = "plotly", color_scheme: str = "modern"):
        """
        Convert business rules text to sophisticated flowchart
        
        Args:
            business_rules_text (str): The business rules text
            process_name (str): Name of the process
            output_format (str): 'plotly', 'graphviz', or 'both'
            color_scheme (str): 'modern' or 'corporate'
        """
        
        # Get structured data from GPT-4
        flowchart_data = self._call_gpt4(business_rules_text, process_name)
        
        # Generate sophisticated visualizations
        if output_format == "plotly":
            self._create_plotly_flowchart(flowchart_data, color_scheme)
        elif output_format == "graphviz":
            self._create_graphviz_flowchart(flowchart_data, color_scheme)
        elif output_format == "both":
            self._create_plotly_flowchart(flowchart_data, color_scheme)
            self._create_graphviz_flowchart(flowchart_data, color_scheme)
        
        return flowchart_data
    
    def _call_gpt4(self, text: str, process_name: str) -> dict:
        """Call custom GPT-4 endpoint to extract flowchart structure"""
        
        prompt = f"""Convert this business rules text to structured flowchart JSON:

REQUIRED JSON FORMAT:
{{
    "title": "{process_name}",
    "description": "Brief process description",
    "nodes": [
        {{
            "id": "unique_id",
            "type": "start|decision|process|filter|end",
            "label": "Short label",
            "description": "Detailed description",
            "technical_details": "Technical implementation details",
            "conditions": ["condition1", "condition2"],
            "actions": ["action1", "action2"],
            "status_codes": {{"success": "200", "error": "500"}},
            "position": {{"x": 0, "y": 0}}
        }}
    ],
    "connections": [
        {{
            "from": "source_id",
            "to": "target_id",
            "label": "condition/action",
            "type": "yes|no|default",
            "description": "Connection description"
        }}
    ],
    "business_rules": [
        {{
            "rule_id": "R001",
            "description": "Rule description",
            "condition": "When condition",
            "action": "Then action",
            "status_codes": ["256", "215"]
        }}
    ]
}}

EXTRACTION RULES:
- Extract ALL conditions as decision nodes
- Extract ALL actions/assignments as process nodes
- Extract ALL filters as filter nodes
- Include exact field names: t_cbms_attribute, grgr_sts, etc.
- Include exact status codes: 256, 215, 216, etc.
- Include exact conditions: attribute_name='GROUP ID FILTER'
- Create separate business rules section
- Add technical implementation details

Business Rules Text:
{text}

Return ONLY valid JSON with complete technical details."""

        try:
            payload = {
                "prompt": prompt,
                "max_tokens": 3000,
                "temperature": 0.2
            }
            
            response = requests.post(self.gpt4_endpoint, json=payload, timeout=45)
            response.raise_for_status()
            
            response_text = response.json().get('response', response.text)
            
            # Extract and parse JSON
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0]
            else:
                json_text = response_text
            
            flowchart_data = json.loads(json_text.strip())
            return self._enhance_data_structure(flowchart_data)
            
        except Exception as e:
            print(f"GPT-4 Error: {e}")
            return self._create_enhanced_fallback(text, process_name)
    
    def _enhance_data_structure(self, data: dict) -> dict:
        """Enhance and validate the flowchart data structure"""
        
        # Ensure all required fields exist
        required_fields = ['title', 'nodes', 'connections']
        for field in required_fields:
            if field not in data:
                data[field] = [] if field != 'title' else 'Business Process'
        
        # Auto-layout nodes using NetworkX
        if data['nodes']:
            data = self._auto_layout_nodes(data)
        
        # Add missing node fields
        for i, node in enumerate(data['nodes']):
            defaults = {
                'id': f'node_{i}',
                'type': 'process',
                'label': f'Step {i+1}',
                'description': 'Process step',
                'technical_details': 'Implementation details',
                'conditions': [],
                'actions': [],
                'status_codes': {},
                'position': {'x': 0, 'y': 0}
            }
            for key, default_value in defaults.items():
                if key not in node:
                    node[key] = default_value
        
        return data
    
    def _auto_layout_nodes(self, data: dict) -> dict:
        """Auto-layout nodes using NetworkX algorithms"""
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in data['nodes']:
            G.add_node(node['id'], **node)
        
        # Add edges
        for conn in data['connections']:
            if conn['from'] in [n['id'] for n in data['nodes']] and conn['to'] in [n['id'] for n in data['nodes']]:
                G.add_edge(conn['from'], conn['to'], **conn)
        
        # Calculate layout using hierarchical positioning
        try:
            # Try hierarchical layout first
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            try:
                # Fallback to spring layout
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                # Manual grid layout as last resort
                pos = self._grid_layout(data['nodes'])
        
        # Update node positions
        for node in data['nodes']:
            if node['id'] in pos:
                x, y = pos[node['id']]
                node['position'] = {'x': float(x), 'y': float(y)}
        
        return data
    
    def _grid_layout(self, nodes: list) -> dict:
        """Simple grid layout as fallback"""
        pos = {}
        cols = int(np.ceil(np.sqrt(len(nodes))))
        for i, node in enumerate(nodes):
            x = (i % cols) * 200
            y = -(i // cols) * 150
            pos[node['id']] = (x, y)
        return pos
    
    def _create_enhanced_fallback(self, text: str, process_name: str) -> dict:
        """Create enhanced fallback structure"""
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        nodes = [{
            'id': 'start',
            'type': 'start',
            'label': 'Start Process',
            'description': f'Begin {process_name}',
            'technical_details': 'Initialize process variables and validate input',
            'conditions': [],
            'actions': ['Initialize'],
            'status_codes': {'start': '100'},
            'position': {'x': 0, 'y': 0}
        }]
        
        connections = []
        business_rules = []
        
        for i, sentence in enumerate(sentences[:5]):
            node_id = f'step_{i}'
            
            # Analyze sentence for type
            if any(word in sentence.lower() for word in ['if', 'check', 'validate', 'when', 'matches']):
                node_type = 'decision'
                conditions = [sentence]
                actions = []
            elif any(word in sentence.lower() for word in ['set', 'assign', 'insert', 'update']):
                node_type = 'process'
                conditions = []
                actions = [sentence]
            else:
                node_type = 'process'
                conditions = []
                actions = [sentence]
            
            nodes.append({
                'id': node_id,
                'type': node_type,
                'label': sentence[:50] + '...' if len(sentence) > 50 else sentence,
                'description': sentence,
                'technical_details': f'Implementation: {sentence}',
                'conditions': conditions,
                'actions': actions,
                'status_codes': {'success': '200', 'error': '400'},
                'position': {'x': 0, 'y': -(i+1)*100}
            })
            
            connections.append({
                'from': 'start' if i == 0 else f'step_{i-1}',
                'to': node_id,
                'label': '',
                'type': 'default',
                'description': f'Flow to step {i+1}'
            })
            
            business_rules.append({
                'rule_id': f'R{i+1:03d}',
                'description': sentence,
                'condition': sentence if node_type == 'decision' else '',
                'action': sentence if node_type == 'process' else '',
                'status_codes': ['200', '400']
            })
        
        # Add end node
        nodes.append({
            'id': 'end',
            'type': 'end',
            'label': 'Process Complete',
            'description': f'End of {process_name}',
            'technical_details': 'Finalize process and return results',
            'conditions': [],
            'actions': ['Complete'],
            'status_codes': {'complete': '200'},
            'position': {'x': 0, 'y': -600}
        })
        
        connections.append({
            'from': f'step_{len(sentences[:5])-1}' if sentences else 'start',
            'to': 'end',
            'label': '',
            'type': 'default',
            'description': 'Complete process'
        })
        
        return {
            'title': process_name,
            'description': f'Automated flowchart for {process_name}',
            'nodes': nodes,
            'connections': connections,
            'business_rules': business_rules
        }
    
    def _create_plotly_flowchart(self, data: dict, color_scheme: str = "modern"):
        """Create interactive Plotly flowchart"""
        
        colors = self.color_schemes[color_scheme]
        
        # Prepare data for Plotly
        node_trace = []
        edge_trace = []
        annotations = []
        
        # Create node positions dictionary
        pos = {node['id']: (node['position']['x'], node['position']['y']) for node in data['nodes']}
        
        # Normalize positions for better display
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Normalize to 0-10 range
            for node_id in pos:
                x, y = pos[node_id]
                if x_max != x_min:
                    x = 10 * (x - x_min) / (x_max - x_min)
                else:
                    x = 5
                if y_max != y_min:
                    y = 10 * (y - y_min) / (y_max - y_min)
                else:
                    y = 5
                pos[node_id] = (x, y)
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for conn in data['connections']:
            if conn['from'] in pos and conn['to'] in pos:
                x0, y0 = pos[conn['from']]
                x1, y1 = pos[conn['to']]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Add arrow annotation
                annotations.append(
                    dict(
                        x=x1, y=y1,
                        ax=x0, ay=y0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        text='',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor='#2C3E50'
                    )
                )
                
                # Add connection label
                if conn.get('label'):
                    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                    label_color = '#00C851' if conn.get('type') == 'yes' else '#FF4444' if conn.get('type') == 'no' else '#2C3E50'
                    
                    annotations.append(
                        dict(
                            x=mid_x, y=mid_y,
                            text=conn['label'],
                            showarrow=False,
                            font=dict(color=label_color, size=10),
                            bgcolor='white',
                            bordercolor=label_color,
                            borderwidth=1
                        )
                    )
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#2C3E50'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
        
        # Create node traces by type
        node_traces = {}
        for node_type in colors.keys():
            node_traces[node_type] = {
                'x': [], 'y': [], 'text': [], 'hovertext': [], 'customdata': []
            }
        
        for node in data['nodes']:
            node_type = node['type']
            if node_type not in node_traces:
                node_type = 'process'  # fallback
            
            x, y = pos.get(node['id'], (5, 5))
            node_traces[node_type]['x'].append(x)
            node_traces[node_type]['y'].append(y)
            node_traces[node_type]['text'].append(node['label'])
            
            # Rich hover text
            hover_text = f"<b>{node['label']}</b><br>"
            hover_text += f"Type: {node_type.title()}<br>"
            hover_text += f"Description: {node.get('description', 'N/A')}<br>"
            hover_text += f"Technical: {node.get('technical_details', 'N/A')}<br>"
            
            if node.get('conditions'):
                hover_text += f"Conditions: {', '.join(node['conditions'])}<br>"
            if node.get('actions'):
                hover_text += f"Actions: {', '.join(node['actions'])}<br>"
            if node.get('status_codes'):
                hover_text += f"Status Codes: {node['status_codes']}"
            
            node_traces[node_type]['hovertext'].append(hover_text)
            node_traces[node_type]['customdata'].append(node)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edge trace
        fig.add_trace(edge_trace)
        
        # Add node traces
        for node_type, trace_data in node_traces.items():
            if trace_data['x']:  # Only add if there are nodes of this type
                color_info = colors[node_type]
                
                # Different symbols for different node types
                symbol_map = {
                    'start': 'circle',
                    'end': 'circle',
                    'decision': 'diamond',
                    'process': 'square',
                    'filter': 'hexagon'
                }
                
                fig.add_trace(go.Scatter(
                    x=trace_data['x'],
                    y=trace_data['y'],
                    mode='markers+text',
                    marker=dict(
                        symbol=symbol_map.get(node_type, 'square'),
                        size=30,
                        color=color_info['bg'],
                        line=dict(width=3, color=color_info['border']),
                        opacity=0.9
                    ),
                    text=trace_data['text'],
                    textposition='middle center',
                    textfont=dict(color=color_info['text'], size=10),
                    hovertext=trace_data['hovertext'],
                    hoverinfo='text',
                    name=node_type.title(),
                    customdata=trace_data['customdata']
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{data.get('title', 'Business Process Flowchart')}</b>",
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='#F8F9FA',
            width=1200,
            height=800
        )
        
        # Add business rules table if available
        if data.get('business_rules'):
            self._add_business_rules_table(fig, data['business_rules'])
        
        # Save and show
        html_file = f"{data.get('title', 'flowchart').replace(' ', '_')}_interactive.html"
        fig.write_html(html_file)
        print(f"🚀 Interactive Plotly flowchart saved: {html_file}")
        
        # Open in browser
        webbrowser.open(f'file://{os.path.abspath(html_file)}')
        
        return fig
    
    def _add_business_rules_table(self, fig, rules):
        """Add business rules table to the figure"""
        
        if not rules:
            return
        
        # Create rules DataFrame
        rules_df = pd.DataFrame(rules)
        
        # Create subplot with table
        from plotly.subplots import make_subplots
        fig_with_table = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Flowchart', 'Business Rules'),
            specs=[[{"type": "scatter"}], [{"type": "table"}]]
        )
        
        # Add all traces from original figure
        for trace in fig.data:
            fig_with_table.add_trace(trace, row=1, col=1)
        
        # Add table
        fig_with_table.add_trace(
            go.Table(
                header=dict(
                    values=list(rules_df.columns),
                    fill_color='#2C3E50',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[rules_df[col] for col in rules_df.columns],
                    fill_color='#F8F9FA',
                    font=dict(color='#2C3E50', size=10),
                    align='left'
                )
            ),
            row=2, col=1
        )
        
        fig_with_table.update_layout(
            title=fig.layout.title,
            height=1200,
            showlegend=True
        )
        
        return fig_with_table
    
    def _create_graphviz_flowchart(self, data: dict, color_scheme: str = "modern"):
        """Create high-quality Graphviz flowchart"""
        
        colors = self.color_schemes[color_scheme]
        
        # Create Graphviz digraph
        dot = Digraph(comment=data.get('title', 'Business Process'))
        dot.attr(rankdir='TB', size='12,16', dpi='300')
        dot.attr('graph', bgcolor='white', fontname='Arial')
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='9')
        
        # Add nodes
        for node in data['nodes']:
            node_type = node['type']
            color_info = colors.get(node_type, colors['process'])
            
            # Set shape based on type
            shape_map = {
                'start': 'ellipse',
                'end': 'ellipse',
                'decision': 'diamond',
                'process': 'box',
                'filter': 'hexagon'
            }
            
            shape = shape_map.get(node_type, 'box')
            
            # Create rich label with HTML-like formatting
            label = f'<{node["label"]}<BR/>'
            if node.get('technical_details') and len(node['technical_details']) < 50:
                label += f'<FONT POINT-SIZE="8">{node["technical_details"]}</FONT><BR/>'
            if node.get('status_codes'):
                status_text = ', '.join([f'{k}:{v}' for k, v in node['status_codes'].items()])
                label += f'<FONT POINT-SIZE="7" COLOR="gray">{status_text}</FONT>'
            label += '>'
            
            dot.node(
                node['id'],
                label=label,
                shape=shape,
                style='filled,rounded',
                fillcolor=color_info['bg'],
                color=color_info['border'],
                fontcolor=color_info['text'],
                width='1.5',
                height='0.8'
            )
        
        # Add edges
        for conn in data['connections']:
            edge_attrs = {
                'color': '#2C3E50',
                'fontcolor': '#2C3E50',
                'penwidth': '2'
            }
            
            if conn.get('label'):
                edge_attrs['label'] = conn['label']
                
                # Color code labels
                if conn.get('type') == 'yes':
                    edge_attrs['color'] = '#00C851'
                    edge_attrs['fontcolor'] = '#00C851'
                elif conn.get('type') == 'no':
                    edge_attrs['color'] = '#FF4444'
                    edge_attrs['fontcolor'] = '#FF4444'
            
            dot.edge(conn['from'], conn['to'], **edge_attrs)
        
        # Render to multiple formats
        base_filename = data.get('title', 'flowchart').replace(' ', '_')
        
        # Save as SVG (vector format)
        dot.render(f'{base_filename}_graphviz', format='svg', cleanup=True)
        print(f"📊 Graphviz SVG saved: {base_filename}_graphviz.svg")
        
        # Save as PNG (high resolution)
        dot.render(f'{base_filename}_graphviz', format='png', cleanup=True)
        print(f"📊 Graphviz PNG saved: {base_filename}_graphviz.png")
        
        # Save as PDF (publication quality)
        try:
            dot.render(f'{base_filename}_graphviz', format='pdf', cleanup=True)
            print(f"📊 Graphviz PDF saved: {base_filename}_graphviz.pdf")
        except:
            print("⚠️  PDF rendering requires additional setup")
        
        return dot


# Sophisticated usage functions
def create_sophisticated_flowchart(text: str, gpt4_endpoint: str, process_name: str = "Business Process",
                                 output_format: str = "both", color_scheme: str = "modern"):
    """
    Create sophisticated flowchart using advanced visualization packages
    
    Args:
        text (str): Business rules text
        gpt4_endpoint (str): Custom GPT-4 endpoint
        process_name (str): Process name
        output_format (str): 'plotly', 'graphviz', or 'both'
        color_scheme (str): 'modern' or 'corporate'
    """
    generator = SophisticatedFlowchartGenerator(gpt4_endpoint)
    return generator.text_to_flowchart(text, process_name, output_format, color_scheme)


def quick_interactive_flowchart(rules_text: str, endpoint_url: str):
    """Quick interactive flowchart using Plotly"""
    generator = SophisticatedFlowchartGenerator(endpoint_url)
    return generator.text_to_flowchart(rules_text, output_format="plotly")


def quick_publication_flowchart(rules_text: str, endpoint_url: str):
    """Quick publication-quality flowchart using Graphviz"""
    generator = SophisticatedFlowchartGenerator(endpoint_url)
    return generator.text_to_flowchart(rules_text, output_format="graphviz")


# Example usage
if __name__ == "__main__":
    
    # Your custom GPT-4 endpoint
    CUSTOM_GPT4_ENDPOINT = "https://your-custom-gpt4-endpoint.com/api/generate"
    
    # Example business rules
    business_rules = """
    The transformation logic for grgr_sts follows these steps:
    
    1. Group ID Filter: Check if HP Group ID (GRGR ID) matches t_cbms_attribute where attribute_name='GROUP ID FILTER'. If matched, set transaction_status='256', record_status='215', insert minimal fields, bypass processing.
    
    2. Benefit ID Filter: Check if HP Benefit ID (PDPD ID) matches t_cbms_attribute where attribute_name='BENEFIT ID FILTER'. If matched, set transaction_status='256', record_status='215', insert minimal fields, bypass processing.
    
    3. Date Validation: Validate Employee Group Effective Date and Term Date. If invalid or missing, set transaction_status='256', record_status='216', set defaults, bypass processing.
    
    4. Field Assignment: Assign grgr_sts based on grgr_term_dt using GroupTermDate logic per TDD Section 3.3.2.
    
    5. Final Validation: Check all processing steps. If successful, complete. If failed, set grgr_sts=NULL.
    """
    
    print("🎨 Creating sophisticated flowcharts...")
    print("📊 Packages: Plotly (interactive) + Graphviz (publication-quality)")
    print("=" * 60)
    
    try:
        # Create both interactive and publication-quality flowcharts
        result = create_sophisticated_flowchart(
            text=business_rules,
            gpt4_endpoint=CUSTOM_GPT4_ENDPOINT,
            process_name="GRGR_STS Transformation Logic",
            output_format="both",
            color_scheme="modern"
        )
        
        print("✅ Sophisticated flowcharts generated successfully!")
        print("📱 Interactive HTML flowchart opened in browser")
        print("🖼️  High-resolution PNG/SVG/PDF files saved")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed: pip install plotly networkx graphviz pandas")


# Installation helper
def install_requirements():
    """Helper function to install required packages"""
    import subprocess
    import sys
    
    packages = [
        'plotly>=5.0.0',
        'networkx>=2.5',
        'graphviz>=0.16',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'requests>=2.25.0'
    ]
    
    print("Installing sophisticated visualization packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
    
    print("\n🎨 Ready to create sophisticated flowcharts!")


if __name__ == "__main__":
    print("Run install_requirements() first if packages are missing")
