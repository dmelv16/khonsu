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
        """
        Detect and tag bus flips in a DataFrame.
        
        :param df: DataFrame with bus monitor data
        :return: DataFrame with 'bus_flip' column added (0=normal, 1=incorrect, 2=correct)
        
        This method is critical for identifying which messages are erroneous due to
        rapid bus transitions that shouldn't occur in normal operation.
        """
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
        """
        Check if two consecutive rows constitute a bus flip.
        
        :param curr: Current row
        :param prev: Previous row
        :param df: Full DataFrame (for column checking)
        :return: True if this is a valid bus flip
        
        Critical validation to ensure we only flag true bus flips and not
        legitimate bus transitions during normal operation.
        """
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
        """
        Validate DC (Data Controller) state requirements.
        
        :param row: Row to check
        :return: True if at least one DC is active
        
        Ensures we only process messages when the system is in a valid operational
        state (at least one DC must be active).
        """
        for col in ['dc1_state', 'dc2_state']:
            if col in row.index and str(row[col]).upper() in ['1', 'TRUE', 'ON', 'YES']:
                return True
        return 'dc1_state' not in row.index and 'dc2_state' not in row.index
    
    @staticmethod
    def _data_changed(r1: pd.Series, r2: pd.Series) -> bool:
        """
        Check if any data word columns changed between two rows.
        
        :param r1: First row
        :param r2: Second row
        :return: True if any data columns differ
        
        Essential check - bus flips only occur when actual data values change.
        If data is identical, it's not a true flip condition.
        """
        cols = [c for c in r1.index if c.startswith('data') and c[4:].isdigit()]
        return any(str(r1[c]) != str(r2[c]) for c in cols if c in r2.index)
    
    @staticmethod
    def _determine_incorrect_row(df: pd.DataFrame, flip_idx: int) -> Tuple[int, int]:
        """
        Determine which row in a flip pair is incorrect.
        
        :param df: Full DataFrame
        :param flip_idx: Index where flip was detected
        :return: (incorrect_idx, correct_idx)
        
        Critical logic to identify which message should be removed. Uses surrounding
        context to determine which bus assignment is the anomaly.
        """
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
    
    Processes data group-by-group (unit_id, station, save), detects flips,
    removes incorrect messages, and generates comprehensive reports.
    """
    
    def __init__(self, parquet_path: str, test_case_dir: str, 
                 requirement_lookup_path: str, output_dir: str = "./output"):
        """
        Initialize the Bus Flip Processor.
        
        :param parquet_path: Path to input parquet file
        :param test_case_dir: Directory containing test case source CSVs
        :param requirement_lookup_path: Path to requirement-testcase lookup CSV
        :param output_dir: Output directory for results
        """
        self.parquet_path = Path(parquet_path)
        self.test_case_dir = Path(test_case_dir)
        self.requirement_lookup_path = Path(requirement_lookup_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cleaned_logs_dir = self.output_dir / "cleaned_logs"
        self.cleaned_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.flip_records = []
        self.test_cases = None
        self.requirement_lookup = None
    
    def run(self):
        """
        Execute the complete bus flip processing pipeline.
        
        Orchestrates the entire workflow: load metadata, process data,
        detect flips, clean data, and generate reports.
        """
        self._load_test_cases()
        self._load_requirement_lookup()
        self._process_parquet()
        self._save_outputs()
    
    def _load_test_cases(self):
        """
        Load all test case source files.
        
        Test cases provide timestamp ranges that map flips to specific test
        executions, enabling test-specific flip analysis.
        
        Handles multiple formats:
        - Single test case: "UYP109_01_Sources.csv" -> test_case_base="UYP109", instance="01"
        - Combined test cases: "UYP109_01&TS2-0043_02_Sources.csv" -> splits into individual test cases
        
        The "_XX" suffix indicates different execution instances of the same test case.
        Combined test cases (using &) are split for requirement lookup but tracked together.
        """
        dfs = []
        for f in self.test_case_dir.glob("*_Sources.csv"):
            df = pd.read_csv(f)
            
            # Remove "_Sources.csv" to get the test case identifier
            full_name = f.stem.replace("_Sources", "")
            
            # Parse the test case name
            # Format: TestCase_XX or TestCase1_XX&TestCase2_XX
            # Extract test case base and instance number
            parts = full_name.rsplit('_', 1)
            if len(parts) == 2:
                test_case_combined = parts[0]
                instance = parts[1]
            else:
                test_case_combined = full_name
                instance = "01"
            
            df['test_case_combined'] = test_case_combined
            df['test_case_instance'] = instance
            df['test_case_full'] = full_name
            
            dfs.append(df)
        
        self.test_cases = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def _load_requirement_lookup(self):
        """
        Load requirement-to-test-case mapping.
        
        Enables linking flips to specific requirements, critical for understanding
        which system requirements may be impacted by bus flip issues.
        
        Expects CSV with columns: "Requirement" and "Test Cases"
        Handles multiple test cases in a single cell (e.g., "TC-001, TC-002, TC-003").
        """
        if not self.requirement_lookup_path.exists():
            self.requirement_lookup = pd.DataFrame(columns=['requirement', 'test_case'])
            return
        
        df = pd.read_csv(self.requirement_lookup_path)
        
        # Verify expected columns exist
        if 'Requirement' not in df.columns or 'Test Cases' not in df.columns:
            print(f"Warning: Expected columns 'Requirement' and 'Test Cases' not found in {self.requirement_lookup_path}")
            self.requirement_lookup = pd.DataFrame(columns=['requirement', 'test_case'])
            return
        
        # Expand rows where multiple test cases are listed
        expanded_rows = []
        for _, row in df.iterrows():
            requirement = row['Requirement']
            test_cases_str = str(row['Test Cases'])
            
            # Split on comma, space, or both
            test_cases = [tc.strip() for tc in test_cases_str.replace(',', ' ').split() if tc.strip()]
            
            for tc in test_cases:
                expanded_rows.append({
                    'requirement': requirement,
                    'test_case': tc
                })
        
        self.requirement_lookup = pd.DataFrame(expanded_rows)
    
    def _process_parquet(self):
        """
        Process parquet data group by group.
        
        Processes each unit_id/station/save combination independently to detect
        flips and generate cleaned output files. Group-wise processing ensures
        flips are detected within correct operational contexts.
        """
        df = pd.read_parquet(self.parquet_path)
        
        for (unit_id, station, save), group in df.groupby(['unit_id', 'station', 'save']):
            processed = self._process_group(group.copy(), unit_id, station, save)
            
            if processed['bus_flip'].isin([1, 2]).any():
                clean_df = processed[processed['bus_flip'] != 1].copy()
                clean_df.loc[clean_df['bus_flip'] == 2, 'bus_flip'] = 0
                clean_df.to_csv(
                    self.output_dir / f"{unit_id}_{station}_{save}_cleaned.csv", 
                    index=False
                )
    
    def _process_group(self, df: pd.DataFrame, unit_id: str, 
                       station: str, save: str) -> pd.DataFrame:
        """
        Process a single group (unit_id, station, save) to detect and record flips.
        
        :param df: Group data
        :param unit_id: Unit identifier
        :param station: Station identifier
        :param save: Save identifier
        :return: DataFrame with bus_flip column
        
        Core processing unit - detects flips and records detailed information
        for later analysis and reporting.
        """
        df = BusFlipDetector.detect_flips(df)
        
        flip_indices = df[df['bus_flip'].isin([1, 2])].index
        for idx in flip_indices:
            if df.loc[idx, 'bus_flip'] == 1:
                self._record_flip(df, idx, unit_id, station, save)
        
        return df
    
    def _record_flip(self, df: pd.DataFrame, flip_idx: int, 
                     unit_id: str, station: str, save: str):
        """
        Record detailed information about a detected flip.
        
        :param df: DataFrame containing the flip
        :param flip_idx: Index of the incorrect message
        :param unit_id: Unit identifier
        :param station: Station identifier
        :param save: Save identifier
        
        Captures all relevant metadata for comprehensive flip tracking and
        analysis, including test case and requirement associations. Also
        records the group-specific index location for precise debugging.
        """
        row = df.iloc[flip_idx]
        msg_type = self._extract_msg_type(row.get('decoded_description', ''))
        test_case = self._find_test_case(row['timestamp'], unit_id, station, save)
        
        correct_idx = flip_idx + 1 if flip_idx + 1 < len(df) and df.loc[flip_idx + 1, 'bus_flip'] == 2 else flip_idx - 1
        correct_bus = df.loc[correct_idx, 'bus'] if 0 <= correct_idx < len(df) else None
        
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
        """
        Extract message type from decoded description.
        
        :param desc: Decoded description string
        :return: Message type or None
        
        Parses message type for categorization and analysis of which
        message types are most affected by flips.
        """
        if pd.isna(desc):
            return None
        match = re.search(r'\[\s*([^\]]+)\s*\]', str(desc))
        return match.group(1) if match else None
    
    def _find_test_case(self, timestamp: float, unit_id: str, 
                        station: str, save: str) -> Optional[str]:
        """
        Find which test case was running at a given timestamp.
        
        :param timestamp: Message timestamp
        :param unit_id: Unit identifier
        :param station: Station identifier
        :param save: Save identifier
        :return: Test case name (combined form without instance suffix) or None
        
        Links flips to specific test executions, enabling test-specific
        analysis and requirement impact assessment. Returns the combined
        test case name (e.g., "UYP109" or "UYP109&TS2-0043") without the
        instance suffix (_01, _02, etc.) for proper requirement lookup.
        """
        if self.test_cases.empty:
            return None
        
        matches = self.test_cases[
            (self.test_cases['unit_id'].astype(str) == str(unit_id)) &
            (self.test_cases['station'].astype(str) == str(station)) &
            (self.test_cases['save'].astype(str) == str(save)) &
            (self.test_cases['timestamp_start'] <= timestamp) &
            (self.test_cases['timestamp_end'] >= timestamp)
        ]
        
        return matches.iloc[0]['test_case_combined'] if not matches.empty else None
    
    def _save_outputs(self):
        """
        Save all output files including parquet and Excel summaries.
        
        Generates comprehensive reports for different analysis perspectives:
        by test case, by location, by message type, and a detailed bus flip index.
        """
        if not self.flip_records:
            return
        
        flips_df = pd.DataFrame(self.flip_records)
        flips_df.to_parquet(self.output_dir / "bus_flips.parquet", index=False)
        
        with pd.ExcelWriter(self.output_dir / "bus_flip_summary.xlsx", engine='openpyxl') as writer:
            self._write_bus_flip_index(flips_df, writer)
            self._write_test_case_summary(flips_df, writer)
            self._write_location_summary(flips_df, writer)
            self._write_msg_type_summary(flips_df, writer)
    
    def _write_bus_flip_index(self, flips_df: pd.DataFrame, writer):
        """
        Write detailed bus flip index to Excel.
        
        :param flips_df: DataFrame of all flip records
        :param writer: Excel writer object
        
        Creates a comprehensive index showing every bus flip with its location
        in the group, message details, and timestamp for precise debugging.
        """
        index_df = flips_df[[
            'unit_id', 'station', 'save', 'group_index', 
            'msg_type', 'decoded_description', 'timestamp',
            'incorrect_bus', 'correct_bus', 'test_case'
        ]].copy()
        
        index_df = index_df.sort_values(['unit_id', 'station', 'save', 'group_index'])
        index_df.to_excel(writer, sheet_name='Bus Flip Index', index=False)
    
    def _write_test_case_summary(self, flips_df: pd.DataFrame, writer):
        """
        Write test case summary sheets to Excel.
        
        :param flips_df: DataFrame of all flip records
        :param writer: Excel writer object
        
        Creates summary by test case with requirement mappings, and a pivot
        showing flips across all location combinations per test case.
        
        Handles combined test cases by splitting them (e.g., "TC-001&TC-002")
        and looking up requirements for each individual test case, then
        combining the requirement lists.
        """
        tc_summary = flips_df.groupby('test_case').agg(
            total_flips=('timestamp', 'count'),
            unique_locations=('unit_id', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index()
        
        if not self.requirement_lookup.empty:
            tc_summary['requirements'] = tc_summary['test_case'].apply(
                lambda tc: self._get_requirements_for_test_case(tc)
            )
        
        tc_summary = tc_summary.sort_values('total_flips', ascending=False)
        tc_summary.to_excel(writer, sheet_name='By Test Case', index=False)
        
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
        """
        Get requirements for a test case, handling combined test cases.
        
        :param test_case: Test case name (may be combined like "TC-001&TC-002")
        :return: Comma-separated list of requirements
        
        Splits combined test cases and looks up requirements for each
        individual test case in the requirement lookup table.
        """
        if pd.isna(test_case):
            return ''
        
        # Split combined test cases on '&' or '&'
        individual_test_cases = [tc.strip() for tc in str(test_case).replace('&', '&').split('&')]
        
        all_requirements = []
        for tc in individual_test_cases:
            reqs = self.requirement_lookup[
                self.requirement_lookup['test_case'] == tc
            ]['requirement'].tolist()
            all_requirements.extend(reqs)
        
        # Remove duplicates and return as comma-separated string
        unique_reqs = list(dict.fromkeys(all_requirements))  # Preserves order
        return ', '.join(unique_reqs) if unique_reqs else ''
    
    def _write_location_summary(self, flips_df: pd.DataFrame, writer):
        """
        Write location-based summary to Excel.
        
        :param flips_df: DataFrame of all flip records
        :param writer: Excel writer object
        
        Shows which unit/station/save combinations have the most flips,
        critical for identifying problematic hardware configurations.
        """
        loc_summary = flips_df.groupby(['unit_id', 'station', 'save']).agg(
            total_flips=('timestamp', 'count'),
            unique_test_cases=('test_case', 'nunique'),
            unique_msg_types=('msg_type', 'nunique')
        ).reset_index().sort_values('total_flips', ascending=False)
        
        loc_summary.to_excel(writer, sheet_name='By Location', index=False)
    
    def _write_msg_type_summary(self, flips_df: pd.DataFrame, writer):
        """
        Write message type summary to Excel.
        
        :param flips_df: DataFrame of all flip records
        :param writer: Excel writer object
        
        Identifies which message types are most susceptible to flips,
        guiding investigation into message-specific issues.
        """
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
