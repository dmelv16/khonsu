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
    Main processor for detecting, removing, and tracking bus flips across CSV data.
    """
    
    def __init__(self, bus_logs_dir: str, test_case_dir: str, 
                 requirement_lookup_path: str, output_dir: str = "./output"):
        self.bus_logs_dir = Path(bus_logs_dir)
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
        self._process_bus_logs()
        self._save_outputs()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
    
    def _parse_bus_log_filename(self, filename: str) -> Optional[dict]:
        """
        Parse bus log filename to extract unit_id, station, save.
        
        Format: unit_id_station_save_rtXX.csv
        Examples: 
            ABC123_L1_1_rt01.csv -> unit_id=ABC123, station=L1, save=1
            ABC123_L1_100_rt01.csv -> unit_id=ABC123, station=L1, save=100
            MY_UNIT_01_L2_56_rt02.csv -> unit_id=MY_UNIT_01, station=L2, save=56
        """
        name = filename.replace('.csv', '').replace('.CSV', '')
        parts = name.rsplit('_', 3)
        
        if len(parts) >= 4:
            unit_id = parts[0]
            station = parts[1]
            save = parts[2]
            rt_suffix = parts[3]
        elif len(parts) == 3:
            if parts[2].lower().startswith('rt'):
                print(f"    WARNING: Unexpected format (no save?): {filename}")
                return None
            unit_id = parts[0]
            station = parts[1]
            save = parts[2]
            rt_suffix = None
        else:
            print(f"    WARNING: Could not parse filename: {filename}")
            return None
        
        return {
            'unit_id': unit_id.strip(),
            'station': station.strip(),
            'save': str(save).strip(),
            'rt_suffix': rt_suffix
        }
    
    def _strip_instance_suffix(self, test_case: str) -> str:
        """
        Remove trailing instance suffix (_01, _02, etc.) from test case name.
        
        Examples:
            UYP1-2_02 -> UYP1-2
            TRTY-0098_01 -> TRTY-0098
            UYP1-2 -> UYP1-2 (unchanged)
        """
        # Only remove _XX where XX is exactly 2 digits at the end
        return re.sub(r'_\d{2}$', '', test_case)
    
    def _load_test_cases(self):
        """
        Load all test case source files.
        
        Handles combined test cases like "UYP1-2_02&TRTY-0098_01_Sources.csv"
        by splitting them into individual test cases for requirement lookup.
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
                
                required_cols = ['timestamp_start', 'timestamp_end', 'unit_id', 'station', 'save']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    print(f"  WARNING: {f.name} missing columns: {missing}")
                    continue
                
                # Parse test case name from filename
                full_name = f.stem.replace("_Sources", "")
                
                # Extract instance number (last _XX where XX is digits)
                parts = full_name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    test_case_combined = parts[0]
                    instance = parts[1]
                else:
                    test_case_combined = full_name
                    instance = "01"
                
                # Split combined test cases on '&' and remove instance suffixes
                # "UYP1-2_02&TRTY-0098_01" -> ["UYP1-2", "TRTY-0098"]
                individual_test_cases = []
                for tc in test_case_combined.split('&'):
                    tc = tc.strip()
                    if tc:
                        tc_clean = self._strip_instance_suffix(tc)
                        individual_test_cases.append(tc_clean)
                
                df['test_case_combined'] = test_case_combined
                df['test_case_instance'] = instance
                df['test_case_full'] = full_name
                df['source_file'] = f.name
                df['individual_test_cases'] = [individual_test_cases] * len(df)
                
                # Normalize identifier columns
                df['unit_id'] = df['unit_id'].astype(str).str.strip()
                df['station'] = df['station'].astype(str).str.strip()
                df['save'] = df['save'].astype(str).str.strip()
                
                dfs.append(df)
                print(f"  Loaded: {f.name}")
                print(f"    -> Combined: {test_case_combined}")
                print(f"    -> Individual test cases: {individual_test_cases}")
                print(f"    -> Rows: {len(df)}")
                
            except Exception as e:
                print(f"  ERROR loading {f.name}: {e}")
        
        if dfs:
            self.test_cases = pd.concat(dfs, ignore_index=True)
            print(f"\n  Total test case entries: {len(self.test_cases)}")
            
            all_individual = set()
            for tc_list in self.test_cases['individual_test_cases']:
                all_individual.update(tc_list)
            print(f"  Unique individual test cases: {sorted(all_individual)}")
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
        
        if not self.requirement_lookup.empty:
            print(f"  Unique requirements: {self.requirement_lookup['requirement'].nunique()}")
            print(f"  Unique test cases in lookup: {self.requirement_lookup['test_case'].nunique()}")
    
    def _diagnose_data(self):
        """Print diagnostic info to help identify matching issues."""
        print("\n[3/5] Diagnosing data compatibility...")
        
        bus_log_files = list(self.bus_logs_dir.glob("*.csv"))
        print(f"\n  BUS LOGS:")
        print(f"    Total CSV files: {len(bus_log_files)}")
        
        bus_log_combos = set()
        for f in bus_log_files[:5]:
            parsed = self._parse_bus_log_filename(f.name)
            if parsed:
                combo = (parsed['unit_id'], parsed['station'], parsed['save'])
                bus_log_combos.add(combo)
                print(f"    Example: {f.name} -> {combo}")
        
        for f in bus_log_files:
            parsed = self._parse_bus_log_filename(f.name)
            if parsed:
                bus_log_combos.add((parsed['unit_id'], parsed['station'], parsed['save']))
        
        print(f"    Unique (unit_id, station, save) combos: {len(bus_log_combos)}")
        
        if self.test_cases is not None and not self.test_cases.empty:
            print("\n  TEST CASE SOURCES:")
            print(f"    Total entries: {len(self.test_cases)}")
            print(f"    Timestamp range: {self.test_cases['timestamp_start'].min():.3f} - {self.test_cases['timestamp_end'].max():.3f}")
            
            tc_combos = set(zip(
                self.test_cases['unit_id'], 
                self.test_cases['station'], 
                self.test_cases['save']
            ))
            
            matching = bus_log_combos & tc_combos
            bus_only = bus_log_combos - tc_combos
            tc_only = tc_combos - bus_log_combos
            
            print(f"\n  MATCHING ANALYSIS:")
            print(f"    Matching combos: {len(matching)}")
            if bus_only:
                print(f"    In bus logs only: {len(bus_only)} combos")
            if tc_only:
                print(f"    In test cases only: {len(tc_only)} combos")
    
    def _process_bus_logs(self):
        """Process all bus log CSVs from the Bus Logs folder."""
        print("\n[4/5] Processing bus log CSVs...")
        
        bus_log_files = list(self.bus_logs_dir.glob("*.csv"))
        
        if not bus_log_files:
            print(f"  ERROR: No CSV files found in {self.bus_logs_dir}")
            return
        
        total_files = len(bus_log_files)
        total_flips = 0
        files_with_flips = 0
        
        print(f"  Found {total_files} CSV files to process")
        
        for idx, csv_file in enumerate(bus_log_files, 1):
            parsed = self._parse_bus_log_filename(csv_file.name)
            if not parsed:
                continue
            
            unit_id = parsed['unit_id']
            station = parsed['station']
            save = parsed['save']
            
            try:
                df = pd.read_csv(csv_file)
                
                if 'timestamp' not in df.columns:
                    print(f"    WARNING: {csv_file.name} missing 'timestamp' column, skipping")
                    continue
                
                processed = self._process_group(df.copy(), unit_id, station, save)
                
                flip_count = (processed['bus_flip'] == 1).sum()
                if flip_count > 0:
                    files_with_flips += 1
                    total_flips += flip_count
                
                output_filename = f"{unit_id}_{station}_{save}_cleaned.csv"
                output_path = self.cleaned_logs_dir / output_filename
                
                if processed['bus_flip'].isin([1, 2]).any():
                    clean_df = processed[processed['bus_flip'] != 1].copy()
                    clean_df.loc[clean_df['bus_flip'] == 2, 'bus_flip'] = 0
                else:
                    clean_df = processed.copy()
                
                clean_df.to_csv(output_path, index=False)
                
            except Exception as e:
                print(f"    ERROR processing {csv_file.name}: {e}")
                continue
            
            if idx % 10 == 0 or idx == total_files:
                print(f"    Processed {idx}/{total_files} files...")
        
        print(f"\n  Summary:")
        print(f"    Total files processed: {total_files}")
        print(f"    Files with flips: {files_with_flips}")
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
        
        test_case_combined, individual_test_cases = self._find_test_case(
            row['timestamp'], unit_id, station, save
        )
        
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
            'test_case_combined': test_case_combined,
            'individual_test_cases': individual_test_cases
        })
    
    def _extract_msg_type(self, desc: str) -> Optional[str]:
        """Extract message type from decoded description."""
        if pd.isna(desc):
            return None
        match = re.search(r'\[\s*([^\]]+)\s*\]', str(desc))
        return match.group(1) if match else None
    
    def _find_test_case(self, timestamp: float, unit_id: str, 
                        station: str, save: str) -> Tuple[Optional[str], list]:
        """Find which test case was running at a given timestamp."""
        if self.test_cases is None or self.test_cases.empty:
            return None, []
        
        matches = self.test_cases[
            (self.test_cases['unit_id'] == str(unit_id)) &
            (self.test_cases['station'] == str(station)) &
            (self.test_cases['save'] == str(save)) &
            (self.test_cases['timestamp_start'] <= timestamp) &
            (self.test_cases['timestamp_end'] >= timestamp)
        ]
        
        if matches.empty:
            return None, []
        
        combined = matches.iloc[0]['test_case_combined']
        individual = matches.iloc[0]['individual_test_cases']
        
        return combined, individual if isinstance(individual, list) else []
    
    def _save_outputs(self):
        """Save all output files including parquet and Excel summaries."""
        print("\n[5/5] Saving outputs...")
        
        if not self.flip_records:
            print("  No flips detected - skipping summary generation")
            return
        
        flips_df = pd.DataFrame(self.flip_records)
        
        flips_df['individual_test_cases_str'] = flips_df['individual_test_cases'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else ''
        )
        
        parquet_path = self.output_dir / "bus_flips.parquet"
        flips_df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path}")
        
        excel_path = self.output_dir / "bus_flip_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self._write_bus_flip_index(flips_df, writer)
            self._write_test_case_summary(flips_df, writer)
            self._write_individual_test_case_summary(flips_df, writer)
            self._write_location_summary(flips_df, writer)
            self._write_msg_type_summary(flips_df, writer)
        print(f"  Saved: {excel_path}")
        
        mapped = flips_df['test_case_combined'].notna().sum()
        unmapped = flips_df['test_case_combined'].isna().sum()
        print(f"\n  Test case mapping:")
        print(f"    Flips mapped to test cases: {mapped}")
        print(f"    Flips without test case: {unmapped}")
    
    def _write_bus_flip_index(self, flips_df: pd.DataFrame, writer):
        """Write detailed bus flip index to Excel."""
        index_df = flips_df[[
            'unit_id', 'station', 'save', 'group_index', 
            'msg_type', 'decoded_description', 'timestamp',
            'incorrect_bus', 'correct_bus', 
            'test_case_combined', 'individual_test_cases_str'
        ]].copy()
        
        index_df = index_df.rename(columns={'individual_test_cases_str': 'individual_test_cases'})
        index_df = index_df.sort_values(['unit_id', 'station', 'save', 'group_index'])
        index_df.to_excel(writer, sheet_name='Bus Flip Index', index=False)
    
    def _write_test_case_summary(self, flips_df: pd.DataFrame, writer):
        """Write combined test case summary to Excel."""
        tc_summary = flips_df.groupby('test_case_combined').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index()
        
        tc_summary = tc_summary.sort_values('total_flips', ascending=False)
        tc_summary.to_excel(writer, sheet_name='By Combined Test Case', index=False)
        
        pivot = flips_df.pivot_table(
            index='test_case_combined',
            columns=['unit_id', 'station', 'save'],
            values='timestamp',
            aggfunc='count',
            fill_value=0
        )
        pivot.columns = ['_'.join(map(str, c)) for c in pivot.columns]
        pivot.reset_index().to_excel(writer, sheet_name='Test Case by Location', index=False)
    
    def _write_individual_test_case_summary(self, flips_df: pd.DataFrame, writer):
        """Write summary by individual test cases (split from combined)."""
        rows = []
        for _, row in flips_df.iterrows():
            individual_tcs = row['individual_test_cases']
            if isinstance(individual_tcs, list) and individual_tcs:
                for tc in individual_tcs:
                    rows.append({
                        'individual_test_case': tc,
                        'unit_id': row['unit_id'],
                        'station': row['station'],
                        'save': row['save'],
                        'msg_type': row['msg_type'],
                        'timestamp': row['timestamp']
                    })
        
        if not rows:
            pd.DataFrame(columns=[
                'individual_test_case', 'total_flips', 'unique_locations', 
                'unique_msg_types', 'requirements'
            ]).to_excel(writer, sheet_name='By Individual Test Case', index=False)
            return
        
        individual_df = pd.DataFrame(rows)
        
        tc_summary = individual_df.groupby('individual_test_case').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index()
        
        if self.requirement_lookup is not None and not self.requirement_lookup.empty:
            tc_summary['requirements'] = tc_summary['individual_test_case'].apply(
                self._get_requirements_for_single_test_case
            )
        
        tc_summary = tc_summary.sort_values('total_flips', ascending=False)
        tc_summary.to_excel(writer, sheet_name='By Individual Test Case', index=False)
    
    def _get_requirements_for_single_test_case(self, test_case: str) -> str:
        """Get requirements for a single test case."""
        if pd.isna(test_case) or test_case == '' or self.requirement_lookup.empty:
            return ''
        
        reqs = self.requirement_lookup[
            self.requirement_lookup['test_case'] == test_case
        ]['requirement'].tolist()
        
        unique_reqs = list(dict.fromkeys(reqs))
        return ', '.join(unique_reqs) if unique_reqs else ''
    
    def _write_location_summary(self, flips_df: pd.DataFrame, writer):
        """Write location-based summary to Excel."""
        loc_summary = flips_df.groupby(['unit_id', 'station', 'save']).agg(
            total_flips=('timestamp', 'count'),
            unique_test_cases=('test_case_combined', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index().sort_values('total_flips', ascending=False)
        
        loc_summary.to_excel(writer, sheet_name='By Location', index=False)
    
    def _write_msg_type_summary(self, flips_df: pd.DataFrame, writer):
        """Write message type summary to Excel."""
        msg_summary = flips_df.groupby('msg_type').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_test_cases=('test_case_combined', 'nunique')
        ).reset_index().sort_values('total_flips', ascending=False)
        
        msg_summary.to_excel(writer, sheet_name='By Message Type', index=False)


if __name__ == "__main__":
    processor = BusFlipProcessor(
        bus_logs_dir="./Bus Logs",
        test_case_dir="./Test Case Sources",
        requirement_lookup_path="./requirement_lookup.csv",
        output_dir="./output"
    )
    processor.run()
