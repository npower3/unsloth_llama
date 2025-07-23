import pandas as pd
import importlib
from tqdm import tqdm
from config.config import llm_scaffold_config
from utils.utils import get_unique_cbms_fields
import os


def get_single_facet_table(base_path, table_name):
    return pd.read_parquet(os.path.join(base_path, f"{table_name}.parquet"))


class Pipeline:
    def __init__(self, input_loc: str, output_loc: str, chunk_size: int = 1000):
        print("Initializing Pipeline...")
        self.df = pd.read_csv(input_loc)
        self.output_loc = output_loc
        self.chunk_size = chunk_size
        self.facet_loc = llm_scaffold_config["facets_loc"]
        self.component_paths = llm_scaffold_config["component_paths"]

        # Preload CBMS mapping
        fields_used_cbms = pd.read_csv(llm_scaffold_config["fields_from_cbms_csv"])
        components_df = pd.read_csv(llm_scaffold_config["components_csv"])
        self.compnt_cbms_map = pd.merge(
            components_df,
            fields_used_cbms,
            left_on="fields",
            right_on="record_id",
            how="left"
        )

        # Preload component classes
        self.component_classes = {
            name: self.import_validation_class(name)
            for name in self.component_paths
        }

        # Precompute CBMS tables needed per component
        self.component_cbms_tables = {
            name: set(get_unique_cbms_fields(name, self.compnt_cbms_map))
            for name in self.component_paths
        }

    def import_validation_class(self, class_name: str):
        module_path = self.component_paths[class_name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def run_flat(self):
        all_results = []

        grouped = self.df.groupby("Employer Group ID")
        group_list = list(grouped)
        total_chunks = (len(group_list) - 1) // self.chunk_size + 1

        for i in range(0, len(group_list), self.chunk_size):
            chunk = group_list[i:i + self.chunk_size]
            print(f"\n▶️ Chunk {i//self.chunk_size + 1}/{total_chunks}")
            chunk_results = []

            for group_id, group_df in tqdm(chunk, desc="Group Processing"):
                records = group_df.to_dict(orient="records")

                # 👇 Load ALL CBMS tables needed for all components
                all_cbms_tables = set().union(*self.component_cbms_tables.values())
                cbms_data = {}
                for tbl in all_cbms_tables:
                    try:
                        cbms_data[tbl] = get_single_facet_table(self.facet_loc, tbl)
                    except Exception as e:
                        print(f"❌ Could not load table {tbl}: {e}")
                        cbms_data[tbl] = pd.DataFrame()

                # 👇 Run all components
                for comp_name, comp_class in self.component_classes.items():
                    try:
                        relevant_tables = {
                            tbl: cbms_data[tbl]
                            for tbl in self.component_cbms_tables[comp_name]
                        }
                        result = comp_class(records, relevant_tables)()

                        if isinstance(result, list):
                            result_df = pd.DataFrame(result)
                        elif isinstance(result, pd.DataFrame):
                            result_df = result
                        else:
                            continue

                        if not result_df.empty:
                            chunk_results.append(result_df)

                    except Exception as e:
                        print(f"❌ {group_id} | {comp_name} failed: {str(e)}")

            if chunk_results:
                chunk_df = pd.concat(chunk_results, ignore_index=True)
                all_results.append(chunk_df)

        # 🔁 Final write
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_pickle(self.output_loc)
            print(f"\n✅ All chunks complete. Output saved to {self.output_loc}")
        else:
            print("⚠️ No results to save.")
