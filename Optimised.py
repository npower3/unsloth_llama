import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time

class Pipeline:
    def __init__(self, input_loc: str, output_loc: str):
        self.df = pd.read_csv(input_loc)
        self.output_loc = output_loc
        self.component_paths = llm_scaffold_config["component_paths"]
        facet_loc = llm_scaffold_config["facets_loc"]
        t1 = time.time()
        self.table_cache = get_facets_tables(facet_loc)
        t2 = time.time()
        print("Time taken to load facets data: ", t2 - t1)
        
        # Optimization 1: Pre-process and index the large table_cache
        self._optimize_table_cache()
        
        # Merge components and fields in cbms
        fields_used_cbms = pd.read_csv(llm_scaffold_config["fields_from_cbms_csv"])
        components_df = pd.read_csv(llm_scaffold_config["components_csv"])
        self.compnt_cbms_map = pd.merge(components_df, fields_used_cbms, 
                                      left_on="fields", right_on="record_id", 
                                      how="left")

    def _optimize_table_cache(self):
        """Pre-process table_cache for faster lookups"""
        if isinstance(self.table_cache, pd.DataFrame):
            # Create indexes for faster filtering
            if 'component_name' in self.table_cache.columns:
                self.table_cache.set_index('component_name', inplace=True, drop=False)
            
            # Convert to dictionary of DataFrames grouped by component for O(1) lookup
            if 'component_name' in self.table_cache.columns:
                self.table_cache_dict = dict(list(self.table_cache.groupby('component_name')))
            else:
                self.table_cache_dict = {'all': self.table_cache}

    def import_validation_class(self, class_name: str):
        module_path = self.component_paths[class_name]
        module = import_module(module_path)
        return getattr(module, class_name)

    def process_single_group(self, group_data):
        """Process a single group - designed for parallel execution"""
        group_id, group_df = group_data
        results = []
        
        print(f"[{group_id}] Starting validation.")
        
        for component_name in tqdm(self.component_paths):
            try:
                # Optimization 2: Use pre-computed dictionary lookup instead of filtering
                if component_name in self.table_cache_dict:
                    cbms_tbls_compnt = self.table_cache_dict[component_name]
                else:
                    continue  # Skip if component not in cache
                
                # Optimization 3: Use vectorized operations for getting unique values
                if len(cbms_tbls_compnt) > 0:
                    cbms_tbls_compnt_values = set(cbms_tbls_compnt.iloc[:, 0].values)  # Faster than to_dict
                else:
                    continue
                
                print(f'Tables from cbms to be used are {len(cbms_tbls_compnt_values)}')
                
                ValidationClass = self.import_validation_class(component_name)
                records = ValidationClass(group_df.to_dict('records'), cbms_tbls_compnt_values)()
                
                if records:  # Only process if there are records
                    records_df = pd.DataFrame(records)
                    results.append(records_df)
                    
            except Exception as e:
                print(f"[{group_id}] {component_name} failed: {str(e)}")
        
        if results:
            combined_results = pd.concat(results, ignore_index=True)
            print(f"[{group_id}] Validation complete.")
            return combined_results
        else:
            return pd.DataFrame()  # Return empty DataFrame if no results

    def run_parallel_processing(self):
        """Run with parallel processing - most effective for CPU-bound tasks"""
        all_results = []
        
        # Split data into chunks for parallel processing
        groups = list(self.df.groupby("Employer Group ID"))
        
        # Determine optimal number of processes (usually CPU count - 1)
        num_processes = min(mp.cpu_count() - 1, len(groups))
        
        print(f"Processing {len(groups)} groups using {num_processes} processes")
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = executor.map(self.process_single_group, groups)
        
        # Collect results
        for result in results:
            if not result.empty:
                all_results.append(result)
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results.to_pickle(self.output_loc)
            print("Processing complete with parallel execution!")
        else:
            print("No results to save.")

    def run_optimized_sequential(self):
        """Optimized sequential version with chunking"""
        all_results = []
        
        # Process in smaller chunks to manage memory
        chunk_size = 1000  # Adjust based on your memory constraints
        groups = list(self.df.groupby("Employer Group ID"))
        
        for i in range(0, len(groups), chunk_size):
            chunk_groups = groups[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(groups)-1)//chunk_size + 1}")
            
            chunk_results = []
            for group_data in tqdm(chunk_groups):
                result = self.process_single_group(group_data)
                if not result.empty:
                    chunk_results.append(result)
            
            if chunk_results:
                chunk_combined = pd.concat(chunk_results, ignore_index=True)
                all_results.append(chunk_combined)
            
            # Optional: Save intermediate results
            # if chunk_results:
            #     chunk_combined.to_pickle(f"{self.output_loc}_chunk_{i//chunk_size}.pickle")
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results.to_pickle(self.output_loc)
            print("Sequential processing complete!")

    def run_with_caching(self):
        """Version with result caching to avoid recomputation"""
        import pickle
        import hashlib
        
        all_results = []
        cache_dir = "validation_cache"  # Create this directory
        
        for group_id, group_df in tqdm(self.df.groupby("Employer Group ID")):
            # Create cache key based on group data
            group_hash = hashlib.md5(pd.util.hash_pandas_object(group_df).values).hexdigest()
            cache_file = f"{cache_dir}/group_{group_id}_{group_hash}.pickle"
            
            try:
                # Try to load from cache
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                    all_results.append(cached_result)
                    print(f"[{group_id}] Loaded from cache")
                    continue
            except FileNotFoundError:
                pass
            
            # Process if not in cache
            result = self.process_single_group((group_id, group_df))
            if not result.empty:
                all_results.append(result)
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results.to_pickle(self.output_loc)

    # Original method kept for compatibility
    def run(self):
        """Choose the best optimization strategy"""
        # For very large datasets, use parallel processing
        if len(self.df) > 50000:
            self.run_parallel_processing()
        else:
            self.run_optimized_sequential()

# Additional optimization utilities
class DataFrameOptimizer:
    @staticmethod
    def reduce_memory_usage(df):
        """Reduce memory usage by optimizing data types"""
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        return df

# Usage example:
# pipeline = Pipeline(input_loc, output_loc)
# pipeline.run()  # This will automatically choose the best optimization strategy
