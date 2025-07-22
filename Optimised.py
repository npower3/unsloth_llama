import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import pickle
import hashlib
import os
import importlib

def import_module(module_path):
    """Import module from path"""
    return importlib.import_module(module_path)

def get_facets_tables(facet_loc):
    """Load facets tables - replace with your actual implementation"""
    # Your existing implementation here
    pass

def get_unique_cbms_fields(component_name, compnt_cbms_map):
    """Get unique CBMS fields for a component"""
    # Your existing implementation here - should return list of table keys
    pass

# Memory optimization removed as requested

class Pipeline:
    def __init__(self, input_loc: str, output_loc: str):
        print("Initializing Pipeline...")
        self.df = pd.read_csv(input_loc)
        self.output_loc = output_loc
        self.component_paths = llm_scaffold_config["component_paths"]
        
        # Load facets data
        facet_loc = llm_scaffold_config["facets_loc"]
        t1 = time.time()
        self.table_cache = get_facets_tables(facet_loc)
        t2 = time.time()
        print("Time taken to load facets data: ", t2 - t1)
        
        # Optimize table_cache immediately after loading
        self._optimize_table_cache()
        
        # Merge components and fields in cbms
        fields_used_cbms = pd.read_csv(llm_scaffold_config["fields_from_cbms_csv"])
        components_df = pd.read_csv(llm_scaffold_config["components_csv"])
        self.compnt_cbms_map = pd.merge(components_df, fields_used_cbms, 
                                      left_on="fields", right_on="record_id", 
                                      how="left")
        
        print("Pipeline initialization complete!")

    def _optimize_table_cache(self):
        """Pre-process table_cache dictionary for faster lookups"""
        print("Optimizing table_cache for faster access...")
        
        # Pre-compute values from each DataFrame in the dictionary
        self.table_cache_values = {}
        
        for key, df in self.table_cache.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Pre-compute the values we'll need during processing
                # Assuming you need values from the first column - adjust as needed
                values = set(df.iloc[:, 0].values)
                self.table_cache_values[key] = values
                
            else:
                self.table_cache_values[key] = set()
        
        print(f"Pre-computed values for {len(self.table_cache_values)} tables")

    def get_optimized_cbms_values(self, component_name):
        """Fast lookup for CBMS values using pre-computed sets"""
        # Get the table keys for this component
        cbms_tbls_compnt = get_unique_cbms_fields(component_name, self.compnt_cbms_map)
        
        # Use pre-computed values instead of iterating through DataFrames
        all_values = set()
        for table_key in cbms_tbls_compnt:
            if table_key in self.table_cache_values:
                all_values.update(self.table_cache_values[table_key])
        
        return all_values

    def import_validation_class(self, class_name: str):
        """Import validation class dynamically"""
        module_path = self.component_paths[class_name]
        module = import_module(module_path)
        return getattr(module, class_name)

    def process_single_group(self, group_data):
        """Process a single group - optimized for parallel execution"""
        group_id, group_df = group_data
        results = []
        
        print(f"[{group_id}] Starting validation.")
        records = group_df.to_dict('records')
        
        for component_name in self.component_paths:
            try:
                # Use optimized lookup
                cbms_tbls_compnt_values = self.get_optimized_cbms_values(component_name)
                
                if not cbms_tbls_compnt_values:
                    continue
                    
                print(f'[{group_id}] {component_name}: Using {len(cbms_tbls_compnt_values)} tables from cbms')
                
                ValidationClass = self.import_validation_class(component_name)
                validation_records = ValidationClass(records, cbms_tbls_compnt_values)()
                
                if validation_records:
                    results.extend(validation_records)
                    
            except Exception as e:
                print(f"[{group_id}] {component_name} failed: {str(e)}")
        
        print(f"[{group_id}] Validation complete.")
        return results

    def run_sequential_optimized(self):
        """Optimized sequential processing"""
        print("Starting sequential optimized processing...")
        all_results = []
        
        for group_id, group_df in tqdm(self.df.groupby("Employer Group ID"), desc="Processing groups"):
            group_results = self.process_single_group((group_id, group_df))
            if group_results:
                all_results.extend(group_results)
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.to_pickle(self.output_loc)
            print(f"Sequential processing complete! Saved {len(final_df)} records.")
        else:
            print("No results to save.")

    def run_parallel_optimized(self, max_workers=None):
        """Optimized parallel processing"""
        groups = list(self.df.groupby("Employer Group ID"))
        
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, len(groups), 4)
        
        print(f"Starting parallel processing with {max_workers} workers...")
        print(f"Processing {len(groups)} groups")
        
        all_results = []
        
        # Use ThreadPoolExecutor for I/O bound tasks, ProcessPoolExecutor for CPU bound
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_group = {executor.submit(self.process_single_group, group): group[0] 
                             for group in groups}
            
            # Collect results as they complete
            for future in tqdm(future_to_group, desc="Collecting results"):
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result)
                except Exception as e:
                    group_id = future_to_group[future]
                    print(f"Group {group_id} failed: {str(e)}")
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.to_pickle(self.output_loc)
            print(f"Parallel processing complete! Saved {len(final_df)} records.")
        else:
            print("No results to save.")

    def run_chunked_processing(self, chunk_size=1000):
        """Process data in chunks to manage memory"""
        print(f"Starting chunked processing with chunk size: {chunk_size}")
        
        groups = list(self.df.groupby("Employer Group ID"))
        all_results = []
        
        for i in range(0, len(groups), chunk_size):
            chunk_groups = groups[i:i+chunk_size]
            chunk_num = i//chunk_size + 1
            total_chunks = (len(groups)-1)//chunk_size + 1
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_groups)} groups)")
            
            chunk_results = []
            for group_data in tqdm(chunk_groups, desc=f"Chunk {chunk_num}"):
                result = self.process_single_group(group_data)
                if result:
                    chunk_results.extend(result)
            
            if chunk_results:
                all_results.extend(chunk_results)
                
                # Optional: Save intermediate results
                if len(all_results) > 50000:  # Save every 50k records
                    temp_df = pd.DataFrame(all_results)
                    temp_df.to_pickle(f"{self.output_loc}_temp_{chunk_num}.pickle")
                    all_results = []  # Clear memory
                    print(f"Saved intermediate results for chunk {chunk_num}")
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.to_pickle(self.output_loc)
            print(f"Chunked processing complete! Saved {len(final_df)} records.")

    def run_with_caching(self, cache_dir="validation_cache"):
        """Run with result caching to avoid recomputation"""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        print(f"Starting processing with caching (cache dir: {cache_dir})")
        all_results = []
        cache_hits = 0
        cache_misses = 0
        
        for group_id, group_df in tqdm(self.df.groupby("Employer Group ID"), desc="Processing groups"):
            # Create cache key based on group data
            group_hash = hashlib.md5(pd.util.hash_pandas_object(group_df).values).hexdigest()
            cache_file = f"{cache_dir}/group_{group_id}_{group_hash}.pickle"
            
            try:
                # Try to load from cache
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                    all_results.extend(cached_result)
                    cache_hits += 1
                    continue
            except FileNotFoundError:
                cache_misses += 1
            
            # Process if not in cache
            result = self.process_single_group((group_id, group_df))
            if result:
                all_results.extend(result)
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
        
        print(f"Cache statistics: {cache_hits} hits, {cache_misses} misses")
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.to_pickle(self.output_loc)
            print(f"Cached processing complete! Saved {len(final_df)} records.")

    def run(self, method='auto', **kwargs):
        """Main run method - automatically chooses best strategy"""
        num_groups = len(self.df.groupby("Employer Group ID"))
        data_size = len(self.df)
        
        print(f"Dataset info: {data_size} rows, {num_groups} groups")
        
        if method == 'auto':
            if data_size > 100000 and num_groups > 50:
                method = 'parallel'
            elif data_size > 50000:
                method = 'chunked'
            else:
                method = 'sequential'
        
        print(f"Using method: {method}")
        
        if method == 'sequential':
            self.run_sequential_optimized()
        elif method == 'parallel':
            self.run_parallel_optimized(**kwargs)
        elif method == 'chunked':
            self.run_chunked_processing(**kwargs)
        elif method == 'cached':
            self.run_with_caching(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_memory_usage_info(self):
        """Get memory usage information"""
        print("Memory usage information:")
        print(f"Main DataFrame: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        total_cache_memory = 0
        for key, df in self.table_cache.items():
            if isinstance(df, pd.DataFrame):
                memory = df.memory_usage(deep=True).sum() / 1024**2
                total_cache_memory += memory
                if memory > 100:  # Show large tables
                    print(f"Table cache '{key}': {memory:.2f} MB")
        
        print(f"Total table cache memory: {total_cache_memory:.2f} MB")

# Example usage:
"""
# Basic usage
pipeline = Pipeline("input.csv", "output.pickle")
pipeline.run()

# Force specific method
pipeline.run(method='parallel', max_workers=6)
pipeline.run(method='chunked', chunk_size=500)
pipeline.run(method='cached', cache_dir='my_cache')

# Check memory usage
pipeline.get_memory_usage_info()
"""
