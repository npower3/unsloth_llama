import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

class SimpleFlowchartGenerator:
    def __init__(self, custom_gpt4_endpoint: str):
        """
        Simple flowchart generator using custom GPT-4 endpoint
        
        Args:
            custom_gpt4_endpoint (str): Your custom GPT-4 API endpoint
        """
        self.gpt4_endpoint = custom_gpt4_endpoint
        self.flowchart_data = None
    
    def text_to_flowchart(self, business_rules_text: str, process_name: str = "Business Process"):
        """
        Convert business rules text to flowchart using GPT-4
        
        Args:
            business_rules_text (str): The business rules text
            process_name (str): Name of the process
        """
        
        # Step 1: Get structured data from GPT-4
        flowchart_data = self._call_gpt4(business_rules_text, process_name)
        
        # Step 2: Create and display flowchart
        self._create_flowchart(flowchart_data)
        
        return flowchart_data
    
    def _call_gpt4(self, text: str, process_name: str) -> dict:
        """Call custom GPT-4 endpoint to convert text to flowchart structure"""
        
        prompt = f"""Convert this business rules text to flowchart JSON format:

REQUIRED JSON FORMAT:
{{
    "title": "{process_name}",
    "nodes": [
        {{"id": "start", "type": "start", "text": "Start", "x": 5, "y": 10}},
        {{"id": "decision1", "type": "decision", "text": "Check condition?", "x": 5, "y": 8}},
        {{"id": "process1", "type": "process", "text": "Do action", "x": 3, "y": 6}},
        {{"id": "end1", "type": "end", "text": "End", "x": 3, "y": 4}}
    ],
    "connections": [
        {{"from": "start", "to": "decision1"}},
        {{"from": "decision1", "to": "process1", "label": "Yes"}},
        {{"from": "decision1", "to": "end1", "label": "No"}}
    ]
}}

RULES:
- Extract ALL conditions as "decision" nodes
- Extract ALL actions as "process" nodes  
- Include exact field names, values, status codes
- Create separate paths for different outcomes
- Position nodes: x=1-9, y=1-10 (top to bottom)

Business Rules Text:
{text}

Return ONLY valid JSON, no other text."""

        try:
            # Call your custom GPT-4 endpoint
            payload = {
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            response = requests.post(self.gpt4_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            # Extract JSON from response
            response_text = response.json().get('response', response.text)
            
            # Clean response to extract JSON
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0]
            else:
                json_text = response_text
            
            # Parse JSON
            flowchart_data = json.loads(json_text.strip())
            
            # Validate structure
            if 'nodes' not in flowchart_data:
                flowchart_data['nodes'] = []
            if 'connections' not in flowchart_data:
                flowchart_data['connections'] = []
            
            return flowchart_data
            
        except Exception as e:
            print(f"GPT-4 Error: {e}")
            # Simple fallback
            return self._create_simple_fallback(text, process_name)
    
    def _create_simple_fallback(self, text: str, process_name: str) -> dict:
        """Create simple flowchart when GPT-4 fails"""
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        nodes = [{"id": "start", "type": "start", "text": "Start", "x": 5, "y": 10}]
        connections = []
        
        y_pos = 8
        prev_id = "start"
        
        for i, sentence in enumerate(sentences[:5]):
            node_id = f"step_{i}"
            
            # Simple classification
            if any(word in sentence.lower() for word in ['if', 'check', 'validate', 'when']):
                node_type = "decision"
            else:
                node_type = "process"
            
            nodes.append({
                "id": node_id,
                "type": node_type, 
                "text": sentence[:60] + "..." if len(sentence) > 60 else sentence,
                "x": 5,
                "y": y_pos
            })
            
            connections.append({"from": prev_id, "to": node_id})
            prev_id = node_id
            y_pos -= 1.5
        
        nodes.append({"id": "end", "type": "end", "text": "End", "x": 5, "y": y_pos})
        connections.append({"from": prev_id, "to": "end"})
        
        return {
            "title": process_name,
            "nodes": nodes,
            "connections": connections
        }
    
    def _create_flowchart(self, data: dict):
        """Create visual flowchart from data"""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 11)
        ax.axis('off')
        
        # Colors
        colors = {
            'start': '#4CAF50',    # Green
            'end': '#F44336',      # Red
            'decision': '#2196F3', # Blue
            'process': '#FF9800'   # Orange
        }
        
        # Draw nodes
        node_positions = {}
        for node in data['nodes']:
            x, y = node['x'], node['y']
            node_positions[node['id']] = (x, y)
            color = colors.get(node['type'], '#CCCCCC')
            
            # Draw shape based on type
            if node['type'] == 'decision':
                # Diamond
                diamond = mpatches.RegularPolygon(
                    (x, y), 4, radius=0.8, orientation=np.pi/4,
                    facecolor=color, edgecolor='black', linewidth=2
                )
                ax.add_patch(diamond)
            elif node['type'] in ['start', 'end']:
                # Oval
                ellipse = mpatches.Ellipse(
                    (x, y), 1.6, 0.8,
                    facecolor=color, edgecolor='black', linewidth=2
                )
                ax.add_patch(ellipse)
            else:
                # Rectangle
                rect = FancyBboxPatch(
                    (x-0.8, y-0.4), 1.6, 0.8,
                    boxstyle="round,pad=0.1",
                    facecolor=color, edgecolor='black', linewidth=2
                )
                ax.add_patch(rect)
            
            # Add text
            ax.text(x, y, node['text'], ha='center', va='center',
                   fontsize=9, weight='bold', wrap=True, color='white')
        
        # Draw connections
        for conn in data['connections']:
            if conn['from'] in node_positions and conn['to'] in node_positions:
                x1, y1 = node_positions[conn['from']]
                x2, y2 = node_positions[conn['to']]
                
                # Arrow
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                
                # Label
                if 'label' in conn and conn['label']:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x + 0.3, mid_y, conn['label'], fontsize=8, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Title
        ax.text(5, 10.5, data.get('title', 'Business Process'), 
               ha='center', va='center', fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Save option
        filename = f"{data.get('title', 'flowchart').replace(' ', '_')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Flowchart saved as: {filename}")


# Simple usage functions
def generate_flowchart(text: str, gpt4_endpoint: str, process_name: str = "Business Process"):
    """
    Simple function to generate flowchart
    
    Args:
        text (str): Business rules text
        gpt4_endpoint (str): Your custom GPT-4 endpoint URL
        process_name (str): Name of the process
    """
    generator = SimpleFlowchartGenerator(gpt4_endpoint)
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
    
    # Generate flowchart
    print("Generating flowchart...")
    
    try:
        flowchart_data = generate_flowchart(
            text=business_rules,
            gpt4_endpoint=CUSTOM_GPT4_ENDPOINT,
            process_name="GRGR_STS Transformation Logic"
        )
        print("Flowchart generated successfully!")
        print(f"Created {len(flowchart_data['nodes'])} nodes and {len(flowchart_data['connections'])} connections")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your GPT-4 endpoint and try again.")


# Alternative: Direct usage with your endpoint
def quick_flowchart(rules_text: str, endpoint_url: str):
    """Ultra-simple one-liner flowchart generation"""
    return SimpleFlowchartGenerator(endpoint_url).text_to_flowchart(rules_text)


# Test with different endpoint formats
def test_endpoint_formats():
    """Test different ways to call your endpoint"""
    
    endpoints = [
        "https://your-domain.com/gpt4/api",
        "https://api.your-service.com/v1/generate",
        "http://localhost:8000/gpt4"
    ]
    
    sample_text = "Check if user exists. If yes, validate password. If valid, login. If not, show error."
    
    for endpoint in endpoints:
        print(f"Testing endpoint: {endpoint}")
        try:
            quick_flowchart(sample_text, endpoint)
            print("✅ Success!")
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
