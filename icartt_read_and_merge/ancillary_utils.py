"""Ancillary utility functions for ICARTT data processing."""

import os
import re
import csv
import pandas as pd
from typing import Dict, List


def read_size_distribution_radii(df: pd.DataFrame, icartt_directory: str) -> Dict[str, List[float]]:
    """
    Read radii information for size distribution variables from ICARTT files.

    This function identifies size distribution instruments in the dataframe,
    searches for their corresponding ICARTT files, and extracts the "Mid Points"
    from the OTHER_COMMENTS header field.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged dataframe from icartt_merger() containing size distribution columns
    icartt_directory : str
        Directory containing the ICARTT files to search
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary mapping instrument names to their radii mid points.
        Example: {'SMPS': [0.01, 0.02, 0.03, ...], 'LAS': [0.1, 0.2, 0.3, ...]}
        
    Raises
    ------
    ValueError
        If mid points differ between files for the same instrument
    FileNotFoundError
        If no files are found for an instrument type
    """
    # Hardcoded list of size distribution instrument types
    size_dist_types = ["LAS", "SMPS"]
    
    # Dictionary to store radii for each instrument
    radii = {}
    
    # Step 1: Check which size distribution types are actually in the dataframe
    df_columns_lower = [col.lower() for col in df.columns]
    found_instruments = []
    
    for inst_type in size_dist_types:
        # Check if any columns contain this instrument type
        # Look for patterns like "LAS_Bin01", "SMPS_Bin01", etc.
        pattern = inst_type.lower()
        matching_cols = [col for col in df_columns_lower if pattern in col and 'bin' in col]
        if matching_cols:
            found_instruments.append(inst_type)
            print(f"Found size distribution instrument: {inst_type}")
        else:
            print(f"Skipping {inst_type} - no matching columns found in dataframe")
    
    if not found_instruments:
        print("Warning: No size distribution instruments found in dataframe")
        return radii
    
    # Step 2-5: Process each found instrument
    for inst_type in found_instruments:
        print(f"\nProcessing {inst_type}...")
        
        # Step 2: Search for files containing "-{instrument}_" pattern
        pattern = f"-{inst_type}_"
        matching_files = []

        for root, dirs, files in os.walk(icartt_directory):
            for f in files:
                if f.endswith('.ict') and pattern in f:
                    filepath = os.path.join(root, f)
                    matching_files.append(filepath)

        if not matching_files:
            print(f"Warning: No files found matching pattern '-{inst_type}_' in {icartt_directory}")
            continue
        
        matching_files = sorted(matching_files)  # Sort for consistency
        print(f"Found {len(matching_files)} file(s) for {inst_type}")
        
        # Step 3: Load first file and extract Mid Points from OTHER_COMMENTS
        first_file = matching_files[0]
        print(f"Reading first file: {os.path.basename(first_file)}")
        
        mid_points = _extract_mid_points(first_file)
        if mid_points is None:
            print(f"Warning: Could not find Mid Points in OTHER_COMMENTS for {inst_type}")
            continue
        
        radii[inst_type] = mid_points
        print(f"Found {len(mid_points)} mid points: {mid_points[:5]}..." if len(mid_points) > 5 else f"Found {len(mid_points)} mid points: {mid_points}")
        
        # Step 4: Check all other files for the same instrument
        for filepath in matching_files[1:]:
            print(f"  Checking {os.path.basename(filepath)}...")
            file_mid_points = _extract_mid_points(filepath)
            
            if file_mid_points is None:
                raise ValueError(f"File {os.path.basename(filepath)} does not contain Mid Points in OTHER_COMMENTS")
            
            # Compare mid points
            if file_mid_points != mid_points:
                raise ValueError(
                    f"Mid Points mismatch for {inst_type}!\n"
                    f"  First file ({os.path.basename(first_file)}): {mid_points}\n"
                    f"  Current file ({os.path.basename(filepath)}): {file_mid_points}"
                )
        
        print(f"✓ All {len(matching_files)} files for {inst_type} have matching mid points")
    
    return radii


def _extract_mid_points(icartt_file: str) -> List[float]:
    """
    Extract Mid Points from the OTHER_COMMENTS header field of an ICARTT file.
    
    Parameters
    ----------
    icartt_file : str
        Path to the ICARTT file
        
    Returns
    -------
    List[float]
        List of mid point radii values, or None if not found
    """
    with open(icartt_file, "r") as f:
        reader = csv.reader(f)
        in_other_comments = False
        mid_points_found = False
        mid_points = []
        
        for row in reader:
            line = " ".join(row).strip()
            
            # Check if we're entering OTHER_COMMENTS section
            if 'OTHER_COMMENTS' in line.upper() or 'OTHER COMMENTS' in line.upper():
                in_other_comments = True
                continue
            
            # If we're in OTHER_COMMENTS, look for "Mid Points" or "MidPoints"
            if in_other_comments:
                # Check if this line contains "Mid Points" or similar
                if 'mid' in line.lower() and 'point' in line.lower():
                    # Try to extract numbers from this line and following lines
                    # Look for pattern like "Mid Points: 0.01, 0.02, 0.03" or similar
                    numbers = re.findall(r'[-+]?\d*\.?\d+', line)
                    if numbers:
                        mid_points = [float(n) for n in numbers]
                        mid_points_found = True
                        break
                
                # Also check if we've hit the end of comments (empty line or new section)
                if not line or line.startswith('PI_CONTACT') or line.startswith('DATA'):
                    break
        
        # If we found mid points, return them
        if mid_points_found and mid_points:
            return mid_points
    
    return None


def ensure_temporal_alignment(df: pd.DataFrame):
    """
    Ensure temporal alignment of all variables in the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to check for temporal alignment
        
    Returns
    -------
    None
    """
    print("NOT COMPLETE")
    return

