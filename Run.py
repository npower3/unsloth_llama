def run(self):
    all_results = []
    
    # Process data in chunks directly without grouping
    total_chunks = len(self.df) // self.chunk_size + (1 if len(self.df) % self.chunk_size else 0)
    
    for i in range(0, len(self.df), self.chunk_size):
        chunk = self.df.iloc[i:i + self.chunk_size]
        print(f"\n Processing chunk {(i//self.chunk_size) + 1}/{total_chunks}")
        
        try:
            # Load ALL CBMS tables needed for all components once per chunk
            all_cbms_tables = set()
            for comp_name in self.component_classes.keys():
                all_cbms_tables.update(self.component_cbms_tables[comp_name])
            
            # Load tables once for the chunk
            cbms_data = {}
            for tbl in all_cbms_tables:
                try:
                    cbms_data[tbl] = get_single_facet_table(self.facet_loc, tbl)
                except Exception as e:
                    print(f" Could not load table {tbl}: {e}")
                    cbms_data[tbl] = pd.DataFrame()
            
            # Process all components for this chunk
            chunk_results = []
            for comp_name, comp_class in self.component_classes.items():
                try:
                    # Get relevant tables for this component
                    relevant_tables = {
                        tbl: cbms_data[tbl] 
                        for tbl in self.component_cbms_tables[comp_name]
                        if tbl in cbms_data and not cbms_data[tbl].empty
                    }
                    
                    if relevant_tables:
                        # Process the chunk with this component
                        result = comp_class(chunk, relevant_tables)()
                        
                        if isinstance(result, list) and result:
                            chunk_results.extend(result)
                        elif isinstance(result, pd.DataFrame) and not result.empty:
                            chunk_results.append(result)
                            
                except Exception as e:
                    print(f" Component {comp_name} failed: {str(e)}")
                    continue
            
            # Combine results from this chunk
            if chunk_results:
                if all(isinstance(r, pd.DataFrame) for r in chunk_results):
                    chunk_df = pd.concat(chunk_results, ignore_index=True)
                else:
                    # Handle mixed result types
                    dfs = []
                    for r in chunk_results:
                        if isinstance(r, pd.DataFrame):
                            dfs.append(r)
                        elif isinstance(r, list):
                            dfs.extend([item for item in r if isinstance(item, pd.DataFrame)])
                    
                    if dfs:
                        chunk_df = pd.concat(dfs, ignore_index=True)
                    else:
                        continue
                
                all_results.append(chunk_df)
                print(f" Chunk processed successfully. Records: {len(chunk_df)}")
            else:
                print(f" No results from chunk {(i//self.chunk_size) + 1}")
                
        except Exception as e:
            print(f" Chunk {(i//self.chunk_size) + 1} failed: {str(e)}")
            continue
    
    # Final write
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(self.output_loc, index=False)  # index=False prevents saving row indices
        print(f"\nAll chunks complete. Output saved to {self.output_loc}")
        print(f"Total records processed: {len(final_df)}")
    else:
        print(" No results to save.")
