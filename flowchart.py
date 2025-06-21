pip install plotly networkx graphviz pandas numpy requests kaleido
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

class BeautifulFlowchartGenerator:
    def __init__(self, custom_gpt4_endpoint: str):
        """
        Beautiful flowchart generator using custom GPT-4 endpoint
        
        Args:
            custom_gpt4_endpoint (str): Your custom GPT-4 API endpoint
        """
        self.gpt4_endpoint = custom_gpt4_endpoint
        self.flowchart_data = None
        
        # Modern color palette
        self.colors = {
            'start': {'bg': '#00C851', 'border': '#007E33', 'text': 'white'},      # Green
            'end': {'bg': '#FF4444', 'border': '#CC0000', 'text': 'white'},        # Red
            'decision': {'bg': '#2E7BFF', 'border': '#0D47A1', 'text': 'white'},   # Blue
            'process': {'bg': '#FF8A00', 'border': '#E65100', 'text': 'white'},    # Orange
            'filter': {'bg': '#AA00FF', 'border': '#6A1B99', 'text': 'white'}      # Purple
        }
    
    def text_to_flowchart(self, business_rules_text: str, process_name: str = "Business Process"):
        """Convert business rules text to beautiful flowchart"""
        
        # Get structured data from GPT-4
        flowchart_data = self._call_gpt4(business_rules_text, process_name)
        
        # Create beautiful flowchart
        self._create_beautiful_flowchart(flowchart_data)
        
        return flowchart_data
    
    def _call_gpt4(self, text: str, process_name: str) -> dict:
        """Call custom GPT-4 endpoint"""
        
        prompt = f"""Convert this business rules text to flowchart JSON format:

REQUIRED JSON FORMAT:
{{
    "title": "{process_name}",
    "nodes": [
        {{"id": "start", "type": "start", "text": "Start Process", "x": 5, "y": 10, "details": "Begin processing"}},
        {{"id": "check1", "type": "decision", "text": "Check Condition?", "x": 5, "y": 8, "details": "Detailed condition description"}},
        {{"id": "action1", "type": "process", "text": "Perform Action", "x": 3, "y": 6, "details": "Action details"}},
        {{"id": "end1", "type": "end", "text": "Complete", "x": 3, "y": 4, "details": "Process complete"}}
    ],
    "connections": [
        {{"from": "start", "to": "check1", "label": ""}},
        {{"from": "check1", "to": "action1", "label": "Yes", "type": "yes"}},
        {{"from": "check1", "to": "end1", "label": "No", "type": "no"}}
    ]
}}

EXTRACT:
- ALL conditions/checks as "decision" nodes
- ALL actions/assignments as "process" nodes  
- ALL filters as "filter" nodes
- Include exact field names, status codes, values
- Add detailed descriptions in "details" field
- Position nodes with x=1-9, y=2-10

Business Rules: {text}

Return ONLY valid JSON."""

        try:
            payload = {"prompt": prompt, "max_tokens": 2000, "temperature": 0.3}
            response = requests.post(self.gpt4_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            response_text = response.json().get('response', response.text)
            
            # Extract JSON
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0]
            else:
                json_text = response_text
            
            flowchart_data = json.loads(json_text.strip())
            return self._validate_data(flowchart_data)
            
        except Exception as e:
            print(f"GPT-4 Error: {e}")
            return self._create_fallback(text, process_name)
    
    def _validate_data(self, data: dict) -> dict:
        """Validate and enhance flowchart data"""
        if 'nodes' not in data:
            data['nodes'] = []
        if 'connections' not in data:
            data['connections'] = []
        
        # Add missing fields
        for node in data['nodes']:
            if 'details' not in node:
                node['details'] = node.get('text', 'No details')
            if 'x' not in node or 'y' not in node:
                node['x'], node['y'] = 5, 5
        
        return data
    
    def _create_fallback(self, text: str, process_name: str) -> dict:
        """Create fallback flowchart"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        nodes = [{
            "id": "start", "type": "start", "text": "Start", 
            "x": 5, "y": 10, "details": f"Begin {process_name}"
        }]
        connections = []
        
        y_pos = 8
        prev_id = "start"
        
        for i, sentence in enumerate(sentences[:4]):
            node_id = f"step_{i}"
            node_type = "decision" if any(word in sentence.lower() for word in ['if', 'check', 'validate']) else "process"
            
            nodes.append({
                "id": node_id, "type": node_type,
                "text": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                "x": 5, "y": y_pos, "details": sentence
            })
            
            connections.append({"from": prev_id, "to": node_id, "label": ""})
            prev_id = node_id
            y_pos -= 1.8
        
        nodes.append({
            "id": "end", "type": "end", "text": "Complete", 
            "x": 5, "y": y_pos, "details": f"End {process_name}"
        })
        connections.append({"from": prev_id, "to": "end", "label": ""})
        
        return {"title": process_name, "nodes": nodes, "connections": connections}
    
    def _create_beautiful_flowchart(self, data: dict):
        """Create beautiful, modern flowchart"""
        
        # Set up figure with better styling
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 11)
        ax.axis('off')
        
        # Add subtle grid
        ax.grid(True, alpha=0.1, color='gray', linestyle='--', linewidth=0.5)
        
        # Draw connections first (behind nodes)
        node_positions = {node['id']: (node['x'], node['y']) for node in data['nodes']}
        self._draw_connections(ax, data['connections'], node_positions)
        
        # Draw nodes
        for node in data['nodes']:
            self._draw_beautiful_node(ax, node)
        
        # Add beautiful title
        self._add_title(ax, data.get('title', 'Business Process'))
        
        # Add legend
        self._add_legend(ax)
        
        # Add footer
        ax.text(5, 0.3, 'Generated by AI • Click nodes for details', 
               ha='center', va='center', fontsize=10, alpha=0.6, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.07)
        plt.show()
        
        # Save high-quality image
        filename = f"{data.get('title', 'flowchart').replace(' ', '_')}_beautiful.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
        print(f"✨ Beautiful flowchart saved: {filename}")
    
    def _draw_beautiful_node(self, ax, node):
        """Draw a beautiful, modern node"""
        x, y = node['x'], node['y']
        node_type = node['type']
        text = node['text']
        
        # Get colors
        colors = self.colors.get(node_type, self.colors['process'])
        
        # Create shadow effect
        shadow_offset = 0.05
        
        if node_type == 'decision':
            # Beautiful diamond with shadow
            shadow = mpatches.RegularPolygon(
                (x + shadow_offset, y - shadow_offset), 4, radius=1.0, 
                orientation=np.pi/4, facecolor='#00000020', edgecolor='none'
            )
            ax.add_patch(shadow)
            
            diamond = mpatches.RegularPolygon(
                (x, y), 4, radius=1.0, orientation=np.pi/4,
                facecolor=colors['bg'], edgecolor=colors['border'], 
                linewidth=3, alpha=0.95
            )
            ax.add_patch(diamond)
            
        elif node_type in ['start', 'end']:
            # Beautiful oval with shadow
            shadow = mpatches.Ellipse(
                (x + shadow_offset, y - shadow_offset), 2.2, 1.0,
                facecolor='#00000020', edgecolor='none'
            )
            ax.add_patch(shadow)
            
            ellipse = mpatches.Ellipse(
                (x, y), 2.2, 1.0,
                facecolor=colors['bg'], edgecolor=colors['border'],
                linewidth=3, alpha=0.95
            )
            ax.add_patch(ellipse)
            
        else:
            # Beautiful rectangle with shadow
            shadow = FancyBboxPatch(
                (x-1.1+shadow_offset, y-0.5-shadow_offset), 2.2, 1.0,
                boxstyle="round,pad=0.1", facecolor='#00000020', edgecolor='none'
            )
            ax.add_patch(shadow)
            
            rect = FancyBboxPatch(
                (x-1.1, y-0.5), 2.2, 1.0,
                boxstyle="round,pad=0.1", facecolor=colors['bg'], 
                edgecolor=colors['border'], linewidth=3, alpha=0.95
            )
            ax.add_patch(rect)
        
        # Add beautiful text with effects
        main_text = ax.text(
            x, y, text, ha='center', va='center',
            fontsize=11, weight='bold', color=colors['text'],
            wrap=True, family='sans-serif'
        )
        main_text.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='#00000030'),
            path_effects.Normal()
        ])
    
    def _draw_connections(self, ax, connections, positions):
        """Draw beautiful connection arrows"""
        
        for conn in connections:
            if conn['from'] not in positions or conn['to'] not in positions:
                continue
                
            x1, y1 = positions[conn['from']]
            x2, y2 = positions[conn['to']]
            
            # Calculate arrow style
            dx, dy = x2 - x1, y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 0.1:
                continue
            
            # Curved arrow for better visual flow
            connection_style = "arc3,rad=0.1" if abs(dx) > 0.5 else "arc3,rad=0"
            
            # Beautiful arrow
            arrow = mpatches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', 
                mutation_scale=25,
                linewidth=2.5,
                color='#2C3E50',
                connectionstyle=connection_style,
                alpha=0.8,
                path_effects=[path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3)]
            )
            ax.add_patch(arrow)
            
            # Add label if present
            label = conn.get('label', '')
            if label:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Offset label to avoid arrow overlap
                offset_x = 0.4 if dx >= 0 else -0.4
                offset_y = 0.2
                
                # Color code labels
                label_color = '#00C851' if conn.get('type') == 'yes' else '#FF4444' if conn.get('type') == 'no' else '#2C3E50'
                
                label_text = ax.text(
                    mid_x + offset_x, mid_y + offset_y, label,
                    fontsize=10, weight='bold', ha='center', va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor='white',
                        edgecolor=label_color,
                        linewidth=2,
                        alpha=0.95
                    ),
                    color=label_color
                )
                label_text.set_path_effects([
                    path_effects.Stroke(linewidth=1, foreground='white'),
                    path_effects.Normal()
                ])
    
    def _add_title(self, ax, title):
        """Add beautiful title"""
        title_text = ax.text(
            5, 10.7, title, ha='center', va='center',
            fontsize=20, weight='bold', color='#2C3E50',
            family='sans-serif'
        )
        title_text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])
        
        # Add subtitle line
        ax.plot([2, 8], [10.4, 10.4], color='#BDC3C7', linewidth=2, alpha=0.7)
    
    def _add_legend(self, ax):
        """Add beautiful legend"""
        legend_elements = []
        legend_labels = []
        
        for node_type, colors in self.colors.items():
            if node_type in ['start', 'decision', 'process', 'end']:
                legend_elements.append(
                    mpatches.Patch(color=colors['bg'], label=node_type.title())
                )
        
        legend = ax.legend(
            legend_elements, [elem.get_label() for elem in legend_elements],
            loc='upper right', bbox_to_anchor=(0.98, 0.98),
            frameon=True, fancybox=True, shadow=True,
            framealpha=0.95, edgecolor='#BDC3C7'
        )
        legend.get_frame().set_facecolor('#FFFFFF')


# Simple usage functions with beautiful output
def generate_beautiful_flowchart(text: str, gpt4_endpoint: str, process_name: str = "Business Process"):
    """Generate a beautiful, professional flowchart"""
    generator = BeautifulFlowchartGenerator(gpt4_endpoint)
    return generator.text_to_flowchart(text, process_name)


# Example usage
if __name__ == "__main__":
    
    # Your custom GPT-4 endpoint
    CUSTOM_GPT4_ENDPOINT = "https://your-custom-gpt4-endpoint.com/api/generate"
    
    # Example business rules
    business_rules = """
    The transformation logic for grgr_sts is as follows:
    
    1. Apply Group ID Filter: Check if HP Group ID (GRGR ID) matches any value in t_cbms_attribute where attribute_name='GROUP ID FILTER'. If matched, set transaction_status='256' and record_status='215', insert minimal fields, and bypass further processing.
    
    2. Apply Benefit ID Filter: Check if HP Benefit ID (PDPD ID) matches any value in t_cbms_attribute where attribute_name='BENEFIT ID FILTER'. If matched, set transaction_status='256' and record_status='215', insert minimal fields, and bypass further processing.
    
    3. Validate Employee Group Dates: Check if Employee Group Effective Date and Term Date are present and valid. If dates are missing or invalid, set transaction_status='256' and record_status='216', set default values, and bypass further processing for errored records.
    
    4. Assign Field Logic: Assign grgr_sts based on grgr_term_dt using GroupTermDate logic. Apply TDD Section 3.3.2 Group File Layout and Mapping Requirements.
    
    5. Final Validation: Validate all filters and validation steps. If all passed, complete processing. If any failed, set grgr_sts = NULL for errored records.
    """
    
    # Generate beautiful flowchart
    print("🎨 Generating beautiful flowchart...")
    
    try:
        flowchart_data = generate_beautiful_flowchart(
            text=business_rules,
            gpt4_endpoint=CUSTOM_GPT4_ENDPOINT,
            process_name="GRGR_STS Transformation Logic"
        )
        print("✨ Beautiful flowchart generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# Ultra-simple beautiful flowchart
def beautiful_flowchart(rules_text: str, endpoint_url: str):
    """One-liner for beautiful flowchart generation"""
    return BeautifulFlowchartGenerator(endpoint_url).text_to_flowchart(rules_text)
