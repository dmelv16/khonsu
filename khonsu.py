#!/usr/bin/env python3
"""
Bus Flip Processor

Detects bus flips, removes incorrect messages, and generates comprehensive tracking reports.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Tuple, Optional, List


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
        """Remove trailing instance suffix (_01, _02, etc.) from test case name."""
        return re.sub(r'_\d{2}$', '', test_case)
    
    def _parse_test_cases_from_filename(self, filename: str) -> List[str]:
        """
        Parse individual test cases from a source filename.
        
        Examples:
            "UYP1-2_02_Sources.csv" -> ["UYP1-2"]
            "UYP1-2_02&TRTY-0098_01_Sources.csv" -> ["UYP1-2", "TRTY-0098"]
        """
        full_name = filename.replace("_Sources.csv", "").replace("_Sources", "")
        
        # Remove the final instance suffix (e.g., _01 at the very end)
        parts = full_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            full_name = parts[0]
        
        # Split on '&' and strip instance suffixes from each
        individual_test_cases = []
        for tc in full_name.split('&'):
            tc = tc.strip()
            if tc:
                tc_clean = self._strip_instance_suffix(tc)
                individual_test_cases.append(tc_clean)
        
        return individual_test_cases
    
    def _load_test_cases(self):
        """Load all test case source files."""
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
                
                # Parse individual test cases from filename
                individual_test_cases = self._parse_test_cases_from_filename(f.name)
                
                # Create a row for each individual test case
                df['source_file'] = f.name
                df['test_cases'] = [individual_test_cases] * len(df)
                
                # Normalize identifier columns
                df['unit_id'] = df['unit_id'].astype(str).str.strip()
                df['station'] = df['station'].astype(str).str.strip()
                df['save'] = df['save'].astype(str).str.strip()
                
                dfs.append(df)
                print(f"  Loaded: {f.name}")
                print(f"    -> Test cases: {individual_test_cases}")
                print(f"    -> Rows: {len(df)}")
                
            except Exception as e:
                print(f"  ERROR loading {f.name}: {e}")
        
        if dfs:
            self.test_cases = pd.concat(dfs, ignore_index=True)
            print(f"\n  Total test case entries: {len(self.test_cases)}")
            
            all_test_cases = set()
            for tc_list in self.test_cases['test_cases']:
                all_test_cases.update(tc_list)
            print(f"  Unique test cases: {sorted(all_test_cases)}")
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
            
            tc_combos = set(zip(
                self.test_cases['unit_id'], 
                self.test_cases['station'], 
                self.test_cases['save']
            ))
            
            matching = bus_log_combos & tc_combos
            print(f"\n  MATCHING ANALYSIS:")
            print(f"    Matching combos: {len(matching)}")
    
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
        
        # Find test cases for this timestamp
        test_cases = self._find_test_cases(row['timestamp'], unit_id, station, save)
        
        correct_idx = None
        correct_bus = None
        
        if flip_idx + 1 < len(df) and df.iloc[flip_idx + 1]['bus_flip'] == 2:
            correct_idx = flip_idx + 1
        elif flip_idx - 1 >= 0 and df.iloc[flip_idx - 1]['bus_flip'] == 2:
            correct_idx = flip_idx - 1
        
        if correct_idx is not None:
            correct_bus = df.iloc[correct_idx]['bus']
        
        # Create a record for each test case (or one with None if no test case)
        if test_cases:
            for tc in test_cases:
                self.flip_records.append({
                    'unit_id': unit_id,
                    'station': station,
                    'save': save,
                    'grouping': f"{unit_id}_{station}_{save}",
                    'msg_type': msg_type,
                    'decoded_description': row.get('decoded_description', ''),
                    'timestamp': row['timestamp'],
                    'group_index': flip_idx,
                    'incorrect_bus': row['bus'],
                    'correct_bus': correct_bus,
                    'test_case': tc
                })
        else:
            self.flip_records.append({
                'unit_id': unit_id,
                'station': station,
                'save': save,
                'grouping': f"{unit_id}_{station}_{save}",
                'msg_type': msg_type,
                'decoded_description': row.get('decoded_description', ''),
                'timestamp': row['timestamp'],
                'group_index': flip_idx,
                'incorrect_bus': row['bus'],
                'correct_bus': correct_bus,
                'test_case': None
            })
    
    def _extract_msg_type(self, desc: str) -> Optional[str]:
        """Extract message type from decoded description."""
        if pd.isna(desc):
            return None
        match = re.search(r'\[\s*([^\]]+)\s*\]', str(desc))
        return match.group(1) if match else None
    
    def _find_test_cases(self, timestamp: float, unit_id: str, 
                         station: str, save: str) -> List[str]:
        """Find which test cases were running at a given timestamp."""
        if self.test_cases is None or self.test_cases.empty:
            return []
        
        matches = self.test_cases[
            (self.test_cases['unit_id'] == str(unit_id)) &
            (self.test_cases['station'] == str(station)) &
            (self.test_cases['save'] == str(save)) &
            (self.test_cases['timestamp_start'] <= timestamp) &
            (self.test_cases['timestamp_end'] >= timestamp)
        ]
        
        if matches.empty:
            return []
        
        # Collect all test cases from matching rows
        all_test_cases = []
        for tc_list in matches['test_cases']:
            if isinstance(tc_list, list):
                all_test_cases.extend(tc_list)
        
        return list(set(all_test_cases))
    
    def _get_requirements(self, test_case: str) -> str:
        """Get requirements for a test case."""
        if pd.isna(test_case) or test_case == '' or self.requirement_lookup is None or self.requirement_lookup.empty:
            return ''
        
        reqs = self.requirement_lookup[
            self.requirement_lookup['test_case'] == test_case
        ]['requirement'].tolist()
        
        unique_reqs = list(dict.fromkeys(reqs))
        return ', '.join(unique_reqs) if unique_reqs else ''
    
    def _save_outputs(self):
        """Save all output files including parquet and Excel summaries."""
        print("\n[5/5] Saving outputs...")
        
        # Collect all groupings from processed files (including those with 0 flips)
        all_groupings = set()
        for f in self.bus_logs_dir.glob("*.csv"):
            parsed = self._parse_bus_log_filename(f.name)
            if parsed:
                all_groupings.add(f"{parsed['unit_id']}_{parsed['station']}_{parsed['save']}")
        
        if not self.flip_records:
            print("  No flips detected")
            flips_df = pd.DataFrame(columns=[
                'unit_id', 'station', 'save', 'grouping', 'msg_type',
                'decoded_description', 'timestamp', 'group_index',
                'incorrect_bus', 'correct_bus', 'test_case'
            ])
        else:
            flips_df = pd.DataFrame(self.flip_records)
        
        parquet_path = self.output_dir / "bus_flips.parquet"
        flips_df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path}")
        
        excel_path = self.output_dir / "bus_flip_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self._write_bus_flip_index(flips_df, writer, all_groupings)
            self._write_test_case_summary(flips_df, writer)
            self._write_location_summary(flips_df, writer)
            self._write_msg_type_summary(flips_df, writer)
        print(f"  Saved: {excel_path}")
        
        mapped = flips_df['test_case'].notna().sum() if len(flips_df) > 0 else 0
        unmapped = flips_df['test_case'].isna().sum() if len(flips_df) > 0 else 0
        print(f"\n  Test case mapping:")
        print(f"    Flips mapped to test cases: {mapped}")
        print(f"    Flips without test case: {unmapped}")
    
    def _write_bus_flip_index(self, flips_df: pd.DataFrame, writer, all_groupings: set):
        """
        Write bus flip index to Excel.
        
        Shows for each grouping:
        - Total flip count (can be 0)
        - List of timestamps where flips occurred
        - Count of A->B and B->A flips
        - Message types affected with counts
        - Index locations matching timestamps
        """
        summary_rows = []
        
        for grouping in sorted(all_groupings):
            grp_flips = flips_df[flips_df['grouping'] == grouping]
            
            flip_count = len(grp_flips)
            
            if flip_count > 0:
                # Get timestamps and indices (deduplicated, sorted)
                grp_unique = grp_flips.drop_duplicates(subset=['timestamp', 'group_index'])
                grp_unique = grp_unique.sort_values('group_index')
                
                timestamps = grp_unique['timestamp'].tolist()
                indices = grp_unique['group_index'].tolist()
                
                # Count A->B and B->A flips
                a_to_b = len(grp_unique[
                    (grp_unique['incorrect_bus'].str.upper() == 'A') & 
                    (grp_unique['correct_bus'].str.upper() == 'B')
                ])
                b_to_a = len(grp_unique[
                    (grp_unique['incorrect_bus'].str.upper() == 'B') & 
                    (grp_unique['correct_bus'].str.upper() == 'A')
                ])
                
                # Message types with counts
                msg_counts = grp_unique['msg_type'].value_counts()
                msg_types_str = ', '.join([
                    f"{msg} ({count})" for msg, count in msg_counts.items() if pd.notna(msg)
                ])
                
                timestamps_str = ', '.join([str(t) for t in timestamps])
                indices_str = ', '.join([str(i) for i in indices])
            else:
                timestamps_str = ''
                indices_str = ''
                a_to_b = 0
                b_to_a = 0
                msg_types_str = ''
            
            summary_rows.append({
                'grouping': grouping,
                'flip_count': flip_count,
                'a_to_b_count': a_to_b,
                'b_to_a_count': b_to_a,
                'message_types': msg_types_str,
                'timestamps': timestamps_str,
                'index_locations': indices_str
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('flip_count', ascending=False)
        summary_df.to_excel(writer, sheet_name='Bus Flip Index', index=False)
    
    def _write_test_case_summary(self, flips_df: pd.DataFrame, writer):
        """
        Write test case summary to Excel.
        
        Shows for each test case:
        - Requirements mapped to it
        - Groupings affected
        - Message types per grouping
        """
        # Filter to only rows with test cases
        tc_df = flips_df[flips_df['test_case'].notna()].copy()
        
        if tc_df.empty:
            pd.DataFrame(columns=[
                'test_case', 'requirements', 'total_flips', 
                'groupings_affected', 'message_types_by_grouping'
            ]).to_excel(writer, sheet_name='By Test Case', index=False)
            return
        
        # Build summary for each test case
        summary_rows = []
        for test_case in tc_df['test_case'].unique():
            tc_data = tc_df[tc_df['test_case'] == test_case]
            
            # Get requirements
            requirements = self._get_requirements(test_case)
            
            # Get groupings affected
            groupings = tc_data['grouping'].unique().tolist()
            
            # Get message types per grouping
            msg_types_by_grouping = {}
            for grouping in groupings:
                grp_data = tc_data[tc_data['grouping'] == grouping]
                msg_types = grp_data['msg_type'].dropna().unique().tolist()
                msg_types_by_grouping[grouping] = msg_types
            
            # Format message types by grouping as string
            msg_types_str = '; '.join([
                f"{grp}: [{', '.join(msgs)}]" 
                for grp, msgs in msg_types_by_grouping.items()
            ])
            
            summary_rows.append({
                'test_case': test_case,
                'requirements': requirements,
                'total_flips': len(tc_data),
                'groupings_affected': ', '.join(groupings),
                'message_types_by_grouping': msg_types_str
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('total_flips', ascending=False)
        summary_df.to_excel(writer, sheet_name='By Test Case', index=False)
    
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
        bus_logs_dir="./Bus Logs",
        test_case_dir="./Test Case Sources",
        requirement_lookup_path="./requirement_lookup.csv",
        output_dir="./output"
    )
    processor.run()
