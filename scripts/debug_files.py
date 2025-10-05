#!/usr/bin/env python3
"""
Debug script to check what files actually exist
"""

import os
from pathlib import Path

def check_sample_files():
    """Check what sample files actually exist"""
    sample_dir = Path("data/sample/images")
    
    print("🔍 Checking sample directory structure...")
    print(f"Sample directory exists: {sample_dir.exists()}")
    
    if sample_dir.exists():
        print(f"Sample directory: {sample_dir.absolute()}")
        all_files = list(sample_dir.glob("*"))
        print(f"Total files in directory: {len(all_files)}")
        
        # Show first 20 files
        print("\n📂 First 20 files:")
        for i, file_path in enumerate(all_files[:20]):
            print(f"  {i+1}. {file_path.name} (exists: {file_path.exists()})")
        
        # Check for specific patterns
        print("\n🔎 Looking for specific file patterns:")
        patterns = ['covid-19_000000.png', 'normal_000000.png', 'viral_pneumonia_000000.png']
        for pattern in patterns:
            matches = list(sample_dir.glob(pattern))
            if matches:
                print(f"  ✅ {pattern}: FOUND - {matches[0].absolute()}")
            else:
                # Try to find similar files
                similar = list(sample_dir.glob(pattern.replace('000000', '*')))
                if similar:
                    print(f"  🔍 {pattern}: NOT FOUND, but similar: {similar[0].name}")
                else:
                    print(f"  ❌ {pattern}: NOT FOUND")
    
    return sample_dir.exists()

def check_actual_files():
    """Check what files we actually have"""
    sample_dir = Path("data/sample/images")
    if not sample_dir.exists():
        print("❌ Sample directory doesn't exist!")
        return
    
    # Get all PNG files
    png_files = list(sample_dir.glob("*.png"))
    if not png_files:
        print("❌ No PNG files found in sample directory!")
        return
    
    print(f"\n🎯 Found {len(png_files)} PNG files")
    
    # Group by class
    classes = {}
    for file_path in png_files:
        filename = file_path.name.lower()
        if 'covid' in filename:
            classes.setdefault('covid-19', []).append(file_path)
        elif 'viral' in filename or 'pneumonia' in filename:
            classes.setdefault('viral_pneumonia', []).append(file_path)
        elif 'normal' in filename:
            classes.setdefault('normal', []).append(file_path)
        else:
            classes.setdefault('unknown', []).append(file_path)
    
    print("\n📊 File distribution by class:")
    for class_name, files in classes.items():
        print(f"  {class_name}: {len(files)} files")
        if files:
            print(f"    Example: {files[0].name}")
    
    return png_files

if __name__ == '__main__':
    print("🛠️ File System Debug Tool")
    print("=" * 50)
    check_sample_files()
    files = check_actual_files()
    
    if files:
        print(f"\n💡 You can use this file for testing: {files[0].absolute()}")