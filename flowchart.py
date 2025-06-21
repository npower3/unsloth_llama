import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import openai
import json
import re
from typing import Dict, List, Any

class BusinessRulesFlowchart:
    def __init__(self, openai_api_key=None):
        """
        Initialize the Business Rules Flowchart generator
        
        Args:
            openai_api_key (str): OpenAI API key for GPT-4 integration
        """
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.fig = None
        self.ax = None
        
    def parse_business_rules(self, rules_text: str) -> Dict[str, Any]:
        """
        Parse the business rules text into structured format
        """
        # Extract key components from the rules text
        structured_rules = {
            "title": "Group STS Transformation Rules",
            "sections": []
        }
        
        # Split into logical sections
        sections = re.split(r'\*\*\*([^*]+)\*\*\*', rules_text)
        
        current_section = None
        for i, section in enumerate(sections):
            if i % 2 == 1:  # Odd indices are section titles
                current_section = {
                    "title": section.strip(),
                    "rules": []
                }
                structured_rules["sections"].append(current_section)
            elif current_section and section.strip():
                # Parse rules within section
                rules = self._extract_rules_from_text(section)
                current_section["rules"].extend(rules)
        
        return structured_rules
    
    def _extract_rules_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual rules from text section"""
        rules = []
        
        # Look for numbered rules or conditional statements
        rule_patterns = [
            r'(\d+)\.\s*([^.]*(?:if|when|where|check)[^.]*\.)',
            r'(Rule \d+[^:]*:)\s*([^.]*\.)',
            r'(-\s*[^-]*(?:if|when|where|check)[^-]*(?=\s*-))',
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    rule = {
                        "id": match[0].strip(),
                        "condition": match[1].strip(),
                        "type": self._classify_rule_type(match[1])
                    }
                    rules.append(rule)
        
        return rules
    
    def _classify_rule_type(self, rule_text: str) -> str:
        """Classify the type of rule based on content"""
        rule_text_lower = rule_text.lower()
        
        if any(word in rule_text_lower for word in ['filter', 'check', 'matches']):
            return 'filter'
        elif any(word in rule_text_lower for word in ['set', 'assign', 'update']):
            return 'assignment'
        elif any(word in rule_text_lower for word in ['validate', 'validation']):
            return 'validation'
        elif any(word in rule_text_lower for word in ['bypass', 'skip']):
            return 'bypass'
        else:
            return 'process'
    
    def convert_with_gpt4(self, rules_text: str) -> Dict[str, Any]:
        """
        Use GPT-4 to convert free text rules to structured format
        """
        try:
            prompt = f"""
            Convert the following business rules text into a structured JSON format suitable for creating a flowchart.
            
            Please structure it as:
            {{
                "title": "Main process title",
                "start_node": "Process start description",
                "decision_nodes": [
                    {{
                        "id": "node_id",
                        "question": "Decision question",
                        "yes_path": "Action if yes",
                        "no_path": "Action if no"
                    }}
                ],
                "process_nodes": [
                    {{
                        "id": "node_id", 
                        "action": "Action description",
                        "next": "next_node_id"
                    }}
                ],
                "end_nodes": ["End condition descriptions"]
            }}
            
            Business Rules Text:
            {rules_text}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing business rules and converting them to structured flowchart data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            structured_data = json.loads(response.choices[0].message.content)
            return structured_data
            
        except Exception as e:
            print(f"GPT-4 conversion failed: {e}")
            # Fallback to manual parsing
            return self.parse_business_rules(rules_text)
    
    def create_flowchart_from_rules(self, rules_text: str, use_gpt4: bool = False):
        """
        Create flowchart from business rules text
        """
        if use_gpt4:
            structured_data = self.convert_with_gpt4(rules_text)
        else:
            structured_data = self._create_flowchart_structure(rules_text)
        
        self._draw_flowchart(structured_data)
    
    def _create_flowchart_structure(self, rules_text: str) -> Dict[str, Any]:
        """
        Create flowchart structure from the specific business rules shown
        """
        return {
            "title": "Group STS Transformation Logic",
            "nodes": [
                {
                    "id": "start",
                    "type": "start",
                    "text": "Start: Process Records",
                    "position": (5, 10)
                },
                {
                    "id": "filter_group_id",
                    "type": "decision",
                    "text": "HP Group ID matches\nt_cbms_attribute?",
                    "position": (5, 8.5)
                },
                {
                    "id": "set_status_256",
                    "type": "process",
                    "text": "Set transaction_status='256'\nrecord_status='215'",
                    "position": (8, 8.5)
                },
                {
                    "id": "filter_benefit_id",
                    "type": "decision", 
                    "text": "HP Benefit ID matches\nt_cbms_attribute?",
                    "position": (5, 7)
                },
                {
                    "id": "validate_dates",
                    "type": "decision",
                    "text": "Valid Effective Date\nand Term Date?",
                    "position": (5, 5.5)
                },
                {
                    "id": "assign_grgr_sts",
                    "type": "process",
                    "text": "Assign grgr_sts based on\ngrgr_term_dt logic",
                    "position": (8, 5.5)
                },
                {
                    "id": "error_handling",
                    "type": "process",
                    "text": "Set grgr_sts = NULL\nfor errored records",
                    "position": (2, 5.5)
                },
                {
                    "id": "final_validation",
                    "type": "decision",
                    "text": "All filters and\nvalidations passed?",
                    "position": (5, 4)
                },
                {
                    "id": "complete",
                    "type": "end",
                    "text": "Process Complete\ngrgr_sts assigned",
                    "position": (8, 4)
                },
                {
                    "id": "bypass",
                    "type": "end", 
                    "text": "Bypass processing\nNo grgr_sts assigned",
                    "position": (2, 4)
                }
            ],
            "connections": [
                ("start", "filter_group_id"),
                ("filter_group_id", "filter_benefit_id", "Yes"),
                ("filter_group_id", "set_status_256", "No"),
                ("set_status_256", "filter_benefit_id"),
                ("filter_benefit_id", "validate_dates", "Yes"),
                ("filter_benefit_id", "error_handling", "No"),
                ("validate_dates", "assign_grgr_sts", "Yes"),
                ("validate_dates", "error_handling", "No"),
                ("assign_grgr_sts", "final_validation"),
                ("error_handling", "final_validation"),
                ("final_validation", "complete", "Yes"),
                ("final_validation", "bypass", "No")
            ]
        }
    
    def _draw_flowchart(self, flowchart_data: Dict[str, Any]):
        """
        Draw the flowchart using matplotlib
        """
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 12))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 11)
        self.ax.axis('off')
        
        # Color scheme
        colors = {
            'start': '#90EE90',      # Light green
            'end': '#FFB6C1',        # Light pink  
            'decision': '#87CEEB',   # Sky blue
            'process': '#F0E68C'     # Khaki
        }
        
        # Draw nodes
        node_positions = {}
        for node in flowchart_data['nodes']:
            x, y = node['position']
            node_positions[node['id']] = (x, y)
            
            # Determine shape and color based on type
            if node['type'] == 'decision':
                # Diamond shape for decisions
                diamond = mpatches.RegularPolygon((x, y), 4, radius=0.8, 
                                                orientation=np.pi/4,
                                                facecolor=colors['decision'],
                                                edgecolor='black', linewidth=2)
                self.ax.add_patch(diamond)
            else:
                # Rectangle for other nodes
                rect_color = colors.get(node['type'], '#FFFFFF')
                if node['type'] == 'start' or node['type'] == 'end':
                    # Rounded rectangle for start/end
                    fancy_box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                                             boxstyle="round,pad=0.1",
                                             facecolor=rect_color,
                                             edgecolor='black', linewidth=2)
                else:
                    # Regular rectangle for process
                    fancy_box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                                             boxstyle="square,pad=0.1", 
                                             facecolor=rect_color,
                                             edgecolor='black', linewidth=2)
                self.ax.add_patch(fancy_box)
            
            # Add text
            self.ax.text(x, y, node['text'], ha='center', va='center',
                        fontsize=9, weight='bold', wrap=True)
        
        # Draw connections
        for connection in flowchart_data['connections']:
            from_node = connection[0]
            to_node = connection[1]
            label = connection[2] if len(connection) > 2 else ""
            
            if from_node in node_positions and to_node in node_positions:
                x1, y1 = node_positions[from_node]
                x2, y2 = node_positions[to_node]
                
                # Draw arrow
                self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                
                # Add label if provided
                if label:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.ax.text(mid_x + 0.2, mid_y + 0.1, label, 
                               fontsize=8, weight='bold', 
                               bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor='white', alpha=0.8))
        
        # Add title
        self.ax.text(5, 10.5, flowchart_data.get('title', 'Business Rules Flowchart'),
                    ha='center', va='center', fontsize=16, weight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=colors['start'], label='Start/End'),
            mpatches.Patch(color=colors['decision'], label='Decision'),
            mpatches.Patch(color=colors['process'], label='Process')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.show()
    
    def save_flowchart(self, filename: str = 'business_rules_flowchart.png'):
        """Save the flowchart to file"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Flowchart saved as {filename}")

# Example usage
def main():
    # Your business rules text from the image
    rules_text = """
    The transformation logic for grgr_sts is as follows: ***Filtering Logic*** (TDD Section 3.3.2 Map2Common Rules - WHA Group, Rule 3; Global Rule #8) 1. Apply Group ID Filter and Benefit ID Filter: - For each record, check if HP Group ID (GRGR ID) matches any value in t_cbms_attribute where attribute_name='GROUP ID FILTER'. - If matched, set transaction_status='256' and record_status='215', insert minimal fields, and bypass further processing. - For each record, check if HP Benefit ID (PDPD ID) matches any value in t_cbms_attribute where attribute_name='BENEFIT ID FILTER'. - If matched, set transaction_status='256' and record_status='215', insert minimal fields, and bypass further processing. ***Validation Steps*** (TDD Section 3.3.2 Map2Common Rules - WHA Group, Rules 4, 5; Global Rule #8) 2. Validate Employee Group Effective Date and Term Date: - If dates are missing, set transaction_status='256' and record_status='216', and set default values. Bypass further processing for errored records. ***Field Assignment Logic*** (TDD Section 3.3.2 Group File Layout and Mapping Requirements; Global Rule #8) 3. Assign grgr_sts based on grgr_term_dt using GroupTermDate logic.
    """
    
    # Initialize the flowchart generator
    flowchart_generator = BusinessRulesFlowchart()
    
    # Option 1: Use manual parsing (no API key required)
    print("Creating flowchart with manual parsing...")
    flowchart_generator.create_flowchart_from_rules(rules_text, use_gpt4=False)
    
    # Option 2: Use GPT-4 (requires API key)
    # Uncomment and add your API key to use GPT-4
    # flowchart_generator = BusinessRulesFlowchart(openai_api_key="your-api-key-here")
    # flowchart_generator.create_flowchart_from_rules(rules_text, use_gpt4=True)
    
    # Save the flowchart
    flowchart_generator.save_flowchart()

if __name__ == "__main__":
    main()
