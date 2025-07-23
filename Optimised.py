import pandas as pd
from tqdm import tqdm
import importlib

class Pipeline:
    def __init__(self, df: pd.DataFrame, component_paths: dict, table_cache: dict, compnt_cbms_map: dict, chunk_size: int = 10, verbose: bool = False):
        self.df = df
        self.chunk_size = chunk_size
        self.component_paths = component_paths
        self.table_cache = table_cache
        self.compnt_cbms_map = compnt_cbms_map
        self.verbose = verbose

        # Preload all validation classes once
        self.validation_classes = {
            comp: self.import_validation_class(comp)
            for comp in self.component_paths
        }

    def import_validation_class(self, class_name: str):
        module_path = self.component_paths[class_name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_unique_cbms_fields(self, component_name: str):
        """Returns list of cbms fields for given component from preloaded map"""
        return self.compnt_cbms_map.get(component_name, [])

    def run(self):
        all_results = []

        # Group data by 'Employer Group ID'
        grouped = list(self.df.groupby("Employer Group ID"))
        total_groups = len(grouped)

        # Chunk of grouped data
        for chunk_start in range(0, total_groups, self.chunk_size):
            chunk_groups = grouped[chunk_start:chunk_start + self.chunk_size]
            if self.verbose:
                print(f"\n🔄 Processing chunk: {chunk_start + 1} to {chunk_start + len(chunk_groups)}")

            for group_id, group_df in chunk_groups:
                try:
                    if self.verbose:
                        print(f"\n✅ Starting validation for Group ID: {group_id}")

                    records = group_df.to_dict(orient='records')

                    for component_name in self.component_paths:
                        try:
                            cbms_keys = self.get_unique_cbms_fields(component_name)
                            cbms_values = {
                                k: self.table_cache[k]
                                for k in cbms_keys if k in self.table_cache
                            }

                            ValidationClass = self.validation_classes[component_name]
                            validator = ValidationClass(records, cbms_values)
                            result = validator.validate() if hasattr(validator, 'validate') else validator  # Optional call
                            all_results.append({
                                "group_id": group_id,
                                "component": component_name,
                                "result": result
                            })

                        except Exception as e:
                            print(f"[⚠️ Component Error] Group: {group_id}, Component: {component_name} ➤ {str(e)}")

                except Exception as e:
                    print(f"[❌ Group Error] Group ID {group_id} ➤ {str(e)}")

        return all_results
