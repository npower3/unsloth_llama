import pandas as pd
from tqdm import tqdm
import importlib
from config.config import llm_scaffold_config
from utils.utils import get_facets_tables, get_unique_cbms_fields


class Pipeline:
    def __init__(self, input_loc: str, output_loc: str, chunk_size: int = 10):
        print("Initializing Pipeline...")
        self.df = pd.read_csv(input_loc)
        self.output_loc = output_loc
        self.chunk_size = chunk_size

        self.component_paths = llm_scaffold_config["component_paths"]
        facet_loc = llm_scaffold_config["facets_loc"]
        self.table_cache = get_facets_tables(facet_loc)

        fields_used_cbms = pd.read_csv(llm_scaffold_config["fields_from_cbms_csv"])
        components_df = pd.read_csv(llm_scaffold_config["components_csv"])

        self.compnt_cbms_map = pd.merge(
            components_df, fields_used_cbms,
            left_on="fields", right_on="record_id", how="left"
        )

    def import_validation_class(self, class_name: str):
        module_path = self.component_paths[class_name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def run(self):
        all_results = []

        group_iterator = list(self.df.groupby("Employer Group ID"))
        total_groups = len(group_iterator)

        for chunk_start in range(0, total_groups, self.chunk_size):
            chunk = group_iterator[chunk_start:chunk_start + self.chunk_size]
            print(f"\n🔄 Processing chunk: {chunk_start + 1} to {chunk_start + len(chunk)}")

            for group_id, group_df in tqdm(chunk, desc="Chunk Progress"):
                print(f"\n▶️ Starting validation for Group ID: {group_id}")
                records = group_df.to_dict(orient='records')

                for component_name in self.component_paths:
                    try:
                        cbms_tbls_compnt = get_unique_cbms_fields(component_name, self.compnt_cbms_map)
                        cbms_keys_set = set(cbms_tbls_compnt)
                        cbms_tbls_compnt_values = {
                            k: self.table_cache[k] for k in cbms_keys_set if k in self.table_cache
                        }

                        ValidationClass = self.import_validation_class(component_name)
                        result_df = ValidationClass(records, cbms_tbls_compnt_values)()

                        if isinstance(result_df, pd.DataFrame):
                            all_results.append(result_df)

                    except Exception as e:
                        print(f"❌ {group_id} | {component_name} failed: {str(e)}")

                print(f"✅ Group ID {group_id} validation complete.")

        final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        final_df.to_csv(self.output_loc, index=False)
        print(f"\n🎉 Validation complete for all chunks. Output saved to {self.output_loc}")
