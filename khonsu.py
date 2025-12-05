#!/usr/bin/env python3
"""
Bus Flip Processor

Detects bus flips, removes incorrect messages, and generates comprehensive tracking reports.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Tuple, Optional


class BusFlipDetector:
    """
    Detects bus flips in bus monitor data.
    
    A bus flip occurs when messages rapidly alternate between Bus A and Bus B
    (within 100ms) for the same message type with changing data values.
    """
    
    FLIP_THRESHOLD_MS = 100
    
    @staticmethod
    def detect_flips(df: pd.DataFrame) -> pd.DataFrame:
        """Detect and tag bus flips in a DataFrame."""
        df = df.copy()
        df['bus_flip'] = 0
        
        if len(df) < 2:
            return df
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(1, len(df)):
            curr, prev = df.iloc[i], df.iloc[i-1]
            
            if not BusFlipDetector._is_flip(curr, prev, df):
                continue
            
            incorrect_idx, correct_idx = BusFlipDetector._determine_incorrect_row(df, i)
            df.loc[incorrect_idx, 'bus_flip'] = 1
            df.loc[correct_idx, 'bus_flip'] = 2
        
        return df
    
    @staticmethod
    def _is_flip(curr: pd.Series, prev: pd.Series, df: pd.DataFrame) -> bool:
        """Check if two consecutive rows constitute a bus flip."""
        if curr['bus'] == prev['bus']:
            return False
        
        if (curr['timestamp'] - prev['timestamp']) * 1000 >= BusFlipDetector.FLIP_THRESHOLD_MS:
            return False
        
        if 'decoded_description' in df.columns:
            if curr['decoded_description'] != prev['decoded_description']:
                return False
        
        if not BusFlipDetector._dc_valid(curr) or not BusFlipDetector._dc_valid(prev):
            return False
        
        if not BusFlipDetector._data_changed(prev, curr):
            return False
        
        return True
    
    @staticmethod
    def _dc_valid(row: pd.Series) -> bool:
        """Validate DC state requirements."""
        for col in ['dc1_state', 'dc2_state']:
            if col in row.index and str(row[col]).upper() in ['1', 'TRUE', 'ON', 'YES']:
                return True
        return 'dc1_state' not in row.index and 'dc2_state' not in row.index
    
    @staticmethod
    def _data_changed(r1: pd.Series, r2: pd.Series) -> bool:
        """Check if any data word columns changed between two rows."""
        cols = [c for c in r1.index if c.startswith('data') and c[4:].isdigit()]
        return any(str(r1[c]) != str(r2[c]) for c in cols if c in r2.index)
    
    @staticmethod
    def _determine_incorrect_row(df: pd.DataFrame, flip_idx: int) -> Tuple[int, int]:
        """Determine which row in a flip pair is incorrect."""
        curr_idx, prev_idx = flip_idx, flip_idx - 1
        curr_bus = df.loc[curr_idx, 'bus']
        prev_bus = df.loc[prev_idx, 'bus']
        
        before_bus = df.loc[prev_idx - 1, 'bus'] if prev_idx > 0 else None
        after_bus = df.loc[curr_idx + 1, 'bus'] if curr_idx < len(df) - 1 else None
        
        if before_bus == curr_bus or after_bus == curr_bus:
            return prev_idx, curr_idx
        if before_bus == prev_bus or after_bus == prev_bus:
            return curr_idx, prev_idx
        
        return curr_idx, prev_idx


class BusFlipProcessor:
    """
    Main processor for detecting, removing, and tracking bus flips across parquet data.
    """
    
    def __init__(self, parquet_path: str, test_case_dir: str, 
                 requirement_lookup_path: str, output_dir: str = "./output"):
        self.parquet_path = Path(parquet_path)
        self.test_case_dir = Path(test_case_dir)
        self.requirement_lookup_path = Path(requirement_lookup_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_logs_dir = self.output_dir / "cleaned_logs"
        self.cleaned_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.flip_records = []
        self.test_cases = None
        self.requirement_lookup = None
    
    def run(self):
        """Execute the complete bus flip processing pipeline."""
        print("=" * 60)
        print("BUS FLIP PROCESSOR")
        print("=" * 60)
        
        self._load_test_cases()
        self._load_requirement_lookup()
        self._diagnose_data()
        self._process_parquet()
        self._save_outputs()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
    
    def _load_test_cases(self):
        """
        Load all test case source files.
        
        Expected columns in source CSVs:
        - timestamp_start: Start time of test case execution
        - timestamp_end: End time of test case execution  
        - unit_id, station, save: Identifiers to match with parquet data
        """
        print("\n[1/5] Loading test case sources...")
        
        dfs = []
        source_files = list(self.test_case_dir.glob("*_Sources.csv"))
        
        if not source_files:
            print(f"  WARNING: No *_Sources.csv files found in {self.test_case_dir}")
            self.test_cases = pd.DataFrame()
            return
        
        for f in source_files:
            try:
                df = pd.read_csv(f)
                
                # Validate required columns
                required_cols = ['timestamp_start', 'timestamp_end', 'unit_id', 'station', 'save']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    print(f"  WARNING: {f.name} missing columns: {missing}")
                    continue
                
                # Parse test case name from filename
                full_name = f.stem.replace("_Sources", "")
                parts = full_name.rsplit('_', 1)
                
                if len(parts) == 2 and parts[1].isdigit():
                    test_case_combined = parts[0]
                    instance = parts[1]
                else:
                    test_case_combined = full_name
                    instance = "01"
                
                df['test_case_combined'] = test_case_combined
                df['test_case_instance'] = instance
                df['test_case_full'] = full_name
                df['source_file'] = f.name
                
                # Normalize identifier columns to strings for matching
                df['unit_id'] = df['unit_id'].astype(str).str.strip()
                df['station'] = df['station'].astype(str).str.strip()
                df['save'] = df['save'].astype(str).str.strip()
                
                dfs.append(df)
                print(f"  Loaded: {f.name} ({len(df)} rows, test_case={test_case_combined})")
                
            except Exception as e:
                print(f"  ERROR loading {f.name}: {e}")
        
        if dfs:
            self.test_cases = pd.concat(dfs, ignore_index=True)
            print(f"  Total test case entries: {len(self.test_cases)}")
        else:
            self.test_cases = pd.DataFrame()
            print("  WARNING: No valid test case data loaded")
    
    def _load_requirement_lookup(self):
        """Load requirement-to-test-case mapping."""
        print("\n[2/5] Loading requirement lookup...")
        
        if not self.requirement_lookup_path.exists():
            print(f"  WARNING: Lookup file not found: {self.requirement_lookup_path}")
            self.requirement_lookup = pd.DataFrame(columns=['requirement', 'test_case'])
            return
        
        df = pd.read_csv(self.requirement_lookup_path)
        
        if 'Requirement' not in df.columns or 'Test Cases' not in df.columns:
            print(f"  WARNING: Expected columns 'Requirement' and 'Test Cases' not found")
            self.requirement_lookup = pd.DataFrame(columns=['requirement', 'test_case'])
            return
        
        expanded_rows = []
        for _, row in df.iterrows():
            requirement = row['Requirement']
            test_cases_str = str(row['Test Cases']).strip()
            
            if pd.isna(test_cases_str) or test_cases_str in ('', 'nan'):
                continue
            
            test_cases = [tc.strip() for tc in test_cases_str.split(',') if tc.strip()]
            for tc in test_cases:
                expanded_rows.append({'requirement': requirement, 'test_case': tc})
        
        self.requirement_lookup = pd.DataFrame(expanded_rows)
        print(f"  Loaded {len(self.requirement_lookup)} requirement-test case mappings")
    
    def _diagnose_data(self):
        """Print diagnostic info to help identify matching issues."""
        print("\n[3/5] Diagnosing data compatibility...")
        
        df = pd.read_parquet(self.parquet_path)
        
        # Normalize parquet identifiers
        df['unit_id'] = df['unit_id'].astype(str).str.strip()
        df['station'] = df['station'].astype(str).str.strip()
        df['save'] = df['save'].astype(str).str.strip()
        
        print("\n  PARQUET DATA:")
        print(f"    Total rows: {len(df)}")
        print(f"    Unique unit_ids: {sorted(df['unit_id'].unique())}")
        print(f"    Unique stations: {sorted(df['station'].unique())}")
        print(f"    Unique saves: {sorted(df['save'].unique())}")
        print(f"    Timestamp range: {df['timestamp'].min():.3f} - {df['timestamp'].max():.3f}")
        
        if not self.test_cases.empty:
            print("\n  TEST CASE SOURCES:")
            print(f"    Total entries: {len(self.test_cases)}")
            print(f"    Unique unit_ids: {sorted(self.test_cases['unit_id'].unique())}")
            print(f"    Unique stations: {sorted(self.test_cases['station'].unique())}")
            print(f"    Unique saves: {sorted(self.test_cases['save'].unique())}")
            print(f"    Timestamp range: {self.test_cases['timestamp_start'].min():.3f} - {self.test_cases['timestamp_end'].max():.3f}")
            print(f"    Test cases: {sorted(self.test_cases['test_case_combined'].unique())}")
            
            # Check for potential mismatches
            parquet_combos = set(zip(df['unit_id'], df['station'], df['save']))
            tc_combos = set(zip(self.test_cases['unit_id'], self.test_cases['station'], self.test_cases['save']))
            
            matching = parquet_combos & tc_combos
            parquet_only = parquet_combos - tc_combos
            tc_only = tc_combos - parquet_combos
            
            print(f"\n  MATCHING ANALYSIS:")
            print(f"    Matching (unit_id, station, save) combos: {len(matching)}")
            if parquet_only:
                print(f"    In parquet but NOT in test cases: {parquet_only}")
            if tc_only:
                print(f"    In test cases but NOT in parquet: {tc_only}")
    
    def _process_parquet(self):
        """Process parquet data group by group, outputting ALL groups."""
        print("\n[4/5] Processing parquet data...")
        
        df = pd.read_parquet(self.parquet_path)
        
        # Normalize identifiers for consistent matching
        df['unit_id'] = df['unit_id'].astype(str).str.strip()
        df['station'] = df['station'].astype(str).str.strip()
        df['save'] = df['save'].astype(str).str.strip()
        
        groups = list(df.groupby(['unit_id', 'station', 'save']))
        total_groups = len(groups)
        total_flips = 0
        groups_with_flips = 0
        
        print(f"  Found {total_groups} groups to process")
        
        for idx, ((unit_id, station, save), group) in enumerate(groups, 1):
            processed = self._process_group(group.copy(), unit_id, station, save)
            
            # Count flips in this group
            flip_count = (processed['bus_flip'] == 1).sum()
            if flip_count > 0:
                groups_with_flips += 1
                total_flips += flip_count
            
            # Always output to cleaned_logs_dir
            output_path = self.cleaned_logs_dir / f"{unit_id}_{station}_{save}_cleaned.csv"
            
            if processed['bus_flip'].isin([1, 2]).any():
                # Remove incorrect messages, reset correct ones
                clean_df = processed[processed['bus_flip'] != 1].copy()
                clean_df.loc[clean_df['bus_flip'] == 2, 'bus_flip'] = 0
            else:
                clean_df = processed.copy()
            
            clean_df.to_csv(output_path, index=False)
            
            # Progress indicator
            if idx % 10 == 0 or idx == total_groups:
                print(f"    Processed {idx}/{total_groups} groups...")
        
        print(f"\n  Summary:")
        print(f"    Total groups processed: {total_groups}")
        print(f"    Groups with flips: {groups_with_flips}")
        print(f"    Total flips detected: {total_flips}")
        print(f"    Cleaned logs saved to: {self.cleaned_logs_dir}")
    
    def _process_group(self, df: pd.DataFrame, unit_id: str, 
                       station: str, save: str) -> pd.DataFrame:
        """Process a single group to detect and record flips."""
        df = BusFlipDetector.detect_flips(df)
        
        flip_indices = df[df['bus_flip'] == 1].index
        for idx in flip_indices:
            self._record_flip(df, idx, unit_id, station, save)
        
        return df
    
    def _record_flip(self, df: pd.DataFrame, flip_idx: int, 
                     unit_id: str, station: str, save: str):
        """Record detailed information about a detected flip."""
        row = df.iloc[flip_idx]
        msg_type = self._extract_msg_type(row.get('decoded_description', ''))
        test_case = self._find_test_case(row['timestamp'], unit_id, station, save)
        
        # Find the correct message in the pair
        correct_idx = None
        correct_bus = None
        
        if flip_idx + 1 < len(df) and df.iloc[flip_idx + 1]['bus_flip'] == 2:
            correct_idx = flip_idx + 1
        elif flip_idx - 1 >= 0 and df.iloc[flip_idx - 1]['bus_flip'] == 2:
            correct_idx = flip_idx - 1
        
        if correct_idx is not None:
            correct_bus = df.iloc[correct_idx]['bus']
        
        self.flip_records.append({
            'unit_id': unit_id,
            'station': station,
            'save': save,
            'msg_type': msg_type,
            'decoded_description': row.get('decoded_description', ''),
            'timestamp': row['timestamp'],
            'group_index': flip_idx,
            'incorrect_bus': row['bus'],
            'correct_bus': correct_bus,
            'test_case': test_case
        })
    
    def _extract_msg_type(self, desc: str) -> Optional[str]:
        """Extract message type from decoded description."""
        if pd.isna(desc):
            return None
        match = re.search(r'\[\s*([^\]]+)\s*\]', str(desc))
        return match.group(1) if match else None
    
    def _find_test_case(self, timestamp: float, unit_id: str, 
                        station: str, save: str) -> Optional[str]:
        """
        Find which test case was running at a given timestamp.
        
        Matches on unit_id, station, save AND checks if timestamp
        falls within the test case's execution window.
        """
        if self.test_cases is None or self.test_cases.empty:
            return None
        
        # Filter for matching location AND timestamp within range
        matches = self.test_cases[
            (self.test_cases['unit_id'] == str(unit_id)) &
            (self.test_cases['station'] == str(station)) &
            (self.test_cases['save'] == str(save)) &
            (self.test_cases['timestamp_start'] <= timestamp) &
            (self.test_cases['timestamp_end'] >= timestamp)
        ]
        
        if matches.empty:
            return None
        
        # Return the combined test case name (without instance suffix)
        return matches.iloc[0]['test_case_combined']
    
    def _save_outputs(self):
        """Save all output files including parquet and Excel summaries."""
        print("\n[5/5] Saving outputs...")
        
        if not self.flip_records:
            print("  No flips detected - skipping summary generation")
            return
        
        flips_df = pd.DataFrame(self.flip_records)
        
        # Save parquet
        parquet_path = self.output_dir / "bus_flips.parquet"
        flips_df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path}")
        
        # Save Excel summary
        excel_path = self.output_dir / "bus_flip_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self._write_bus_flip_index(flips_df, writer)
            self._write_test_case_summary(flips_df, writer)
            self._write_location_summary(flips_df, writer)
            self._write_msg_type_summary(flips_df, writer)
        print(f"  Saved: {excel_path}")
        
        # Print test case mapping summary
        if 'test_case' in flips_df.columns:
            mapped = flips_df['test_case'].notna().sum()
            unmapped = flips_df['test_case'].isna().sum()
            print(f"\n  Test case mapping:")
            print(f"    Flips mapped to test cases: {mapped}")
            print(f"    Flips without test case: {unmapped}")
            
            if mapped > 0:
                print(f"    Test cases with flips: {flips_df['test_case'].dropna().unique().tolist()}")
    
    def _write_bus_flip_index(self, flips_df: pd.DataFrame, writer):
        """Write detailed bus flip index to Excel."""
        index_df = flips_df[[
            'unit_id', 'station', 'save', 'group_index', 
            'msg_type', 'decoded_description', 'timestamp',
            'incorrect_bus', 'correct_bus', 'test_case'
        ]].copy()
        
        index_df = index_df.sort_values(['unit_id', 'station', 'save', 'group_index'])
        index_df.to_excel(writer, sheet_name='Bus Flip Index', index=False)
    
    def _write_test_case_summary(self, flips_df: pd.DataFrame, writer):
        """Write test case summary sheets to Excel."""
        tc_summary = flips_df.groupby('test_case').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index()
        
        if not self.requirement_lookup.empty:
            tc_summary['requirements'] = tc_summary['test_case'].apply(
                self._get_requirements_for_test_case
            )
        
        tc_summary = tc_summary.sort_values('total_flips', ascending=False)
        tc_summary.to_excel(writer, sheet_name='By Test Case', index=False)
        
        # Pivot table
        pivot = flips_df.pivot_table(
            index='test_case',
            columns=['unit_id', 'station', 'save'],
            values='timestamp',
            aggfunc='count',
            fill_value=0
        )
        pivot.columns = ['_'.join(map(str, c)) for c in pivot.columns]
        pivot.reset_index().to_excel(writer, sheet_name='Test Case by Location', index=False)
    
    def _get_requirements_for_test_case(self, test_case: str) -> str:
        """Get requirements for a test case, handling combined test cases."""
        if pd.isna(test_case) or test_case == '':
            return ''
        
        individual_tcs = [tc.strip() for tc in str(test_case).split('&') if tc.strip()]
        
        all_reqs = []
        for tc in individual_tcs:
            reqs = self.requirement_lookup[
                self.requirement_lookup['test_case'] == tc
            ]['requirement'].tolist()
            all_reqs.extend(reqs)
        
        unique_reqs = list(dict.fromkeys(all_reqs))
        return ', '.join(unique_reqs) if unique_reqs else ''
    
    def _write_location_summary(self, flips_df: pd.DataFrame, writer):
        """Write location-based summary to Excel."""
        loc_summary = flips_df.groupby(['unit_id', 'station', 'save']).agg(
            total_flips=('timestamp', 'count'),
            unique_test_cases=('test_case', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index().sort_values('total_flips', ascending=False)
        
        loc_summary.to_excel(writer, sheet_name='By Location', index=False)
    
    def _write_msg_type_summary(self, flips_df: pd.DataFrame, writer):
        """Write message type summary to Excel."""
        msg_summary = flips_df.groupby('msg_type').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_test_cases=('test_case', 'nunique')
        ).reset_index().sort_values('total_flips', ascending=False)
        
        msg_summary.to_excel(writer, sheet_name='By Message Type', index=False)


if __name__ == "__main__":
    processor = BusFlipProcessor(
        parquet_path="bus_monitor_data.parquet",
        test_case_dir="./Test Case Sources",
        requirement_lookup_path="./requirement_lookup.csv",
        output_dir="./output"
    )
    processor.run()
