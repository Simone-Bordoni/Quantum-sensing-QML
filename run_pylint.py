#!/usr/bin/env python3
"""
Pylint Runner and Auto-Fixer Script for Quantum Sensing Optimization Library
===========================================================================

This script runs Pylint on the qsopt library and applies automatic fixes where possible.
It combines Pylint for analysis with autopep8 and black for automatic formatting fixes.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


class PylintRunner:
    """Run Pylint analysis and apply automatic fixes."""
    
    def __init__(self, source_dir: str = "src/qsopt", config_file: str = ".pylintrc"):
        """
        Initialize Pylint runner.
        
        Args:
            source_dir: Directory containing source code to analyze
            config_file: Path to pylint configuration file
        """
        self.source_dir = Path(source_dir)
        self.config_file = Path(config_file)
        self.results = {}
        
    def run_pylint_analysis(self, output_format: str = "json") -> Any:
        """
        Run Pylint analysis on the source directory.
        
        Args:
            output_format: Output format for pylint (json, text, parseable)
            
        Returns:
            Pylint results (list for json format, None for text format)
        """
        print(f"🔍 Running Pylint analysis on {self.source_dir}")
        
        # Construct pylint command
        cmd = [
            sys.executable, "-m", "pylint",
            "--rcfile", str(self.config_file),
            "--output-format", output_format,
            str(self.source_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if output_format == "json":
                try:
                    self.results = json.loads(result.stdout) if result.stdout.strip() else []
                except json.JSONDecodeError:
                    print("⚠️  Warning: Could not parse JSON output from Pylint")
                    self.results = []
            else:
                print(result.stdout)
                
            # Print summary
            print(f"📊 Pylint analysis completed with return code: {result.returncode}")
            if isinstance(self.results, list):
                print(f"📋 Found {len(self.results)} issues to review")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error running Pylint: {e}")
            return {}
    
    def apply_autopep8_fixes(self) -> None:
        """Apply PEP 8 formatting fixes using autopep8."""
        print("🔧 Applying PEP 8 fixes with autopep8...")
        
        cmd = [
            sys.executable, "-m", "autopep8",
            "--in-place",
            "--recursive",
            "--aggressive",
            "--aggressive",
            str(self.source_dir)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ autopep8 fixes applied successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error applying autopep8 fixes: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
    
    def apply_black_formatting(self) -> None:
        """Apply Black code formatting."""
        print("🎨 Applying Black formatting...")
        
        cmd = [
            sys.executable, "-m", "black",
            "--line-length", "100",
            str(self.source_dir)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Black formatting applied successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error applying Black formatting: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
    
    def analyze_results(self) -> None:
        """Analyze and categorize Pylint results."""
        if not isinstance(self.results, list) or not self.results:
            print("📝 No issues found or no results to analyze")
            return
            
        # Categorize issues
        categories = {}
        for issue in self.results:
            category = issue.get('type', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(issue)
        
        print("\n📊 Issue Summary:")
        print("=" * 50)
        
        for category, issues in categories.items():
            print(f"{category.upper()}: {len(issues)} issues")
            
            # Show first few examples
            for i, issue in enumerate(issues[:3]):
                file_path = issue.get('path', 'unknown')
                line = issue.get('line', '?')
                message = issue.get('message', 'No message')
                msg_id = issue.get('message-id', 'unknown')
                
                print(f"  {i+1}. {file_path}:{line} - {msg_id}: {message}")
            
            if len(issues) > 3:
                print(f"  ... and {len(issues) - 3} more")
            print()
    
    def generate_report(self, output_file: str = "pylint_report.txt") -> None:
        """Generate a detailed report of Pylint results."""
        if not isinstance(self.results, list):
            return
            
        report_path = Path(output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Pylint Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            if not self.results:
                f.write("No issues found! 🎉\n")
                return
            
            # Group by file
            files = {}
            for issue in self.results:
                file_path = issue.get('path', 'unknown')
                if file_path not in files:
                    files[file_path] = []
                files[file_path].append(issue)
            
            for file_path, issues in files.items():
                f.write(f"\nFile: {file_path}\n")
                f.write("-" * (len(file_path) + 6) + "\n")
                
                for issue in issues:
                    line = issue.get('line', '?')
                    column = issue.get('column', '?')
                    msg_id = issue.get('message-id', 'unknown')
                    message = issue.get('message', 'No message')
                    issue_type = issue.get('type', 'unknown')
                    
                    f.write(f"  {line}:{column} {issue_type.upper()} {msg_id}: {message}\n")
        
        print(f"📄 Detailed report saved to {report_path}")


def main():
    """Main entry point for the Pylint runner script."""
    parser = argparse.ArgumentParser(
        description="Run Pylint analysis and apply automatic fixes"
    )
    parser.add_argument(
        "--source-dir", "-s",
        default="src/qsopt",
        help="Source directory to analyze (default: src/qsopt)"
    )
    parser.add_argument(
        "--config", "-c",
        default=".pylintrc",
        help="Pylint configuration file (default: .pylintrc)"
    )
    parser.add_argument(
        "--fix", "-f",
        action="store_true",
        help="Apply automatic fixes (autopep8 + black)"
    )
    parser.add_argument(
        "--report", "-r",
        default="pylint_report.txt",
        help="Output file for detailed report (default: pylint_report.txt)"
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "text", "parseable"],
        help="Pylint output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not Path(args.source_dir).exists():
        print(f"❌ Source directory {args.source_dir} does not exist")
        sys.exit(1)
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Config file {args.config} does not exist")
        sys.exit(1)
    
    print("🚀 Starting Pylint Analysis and Auto-Fix Process")
    print("=" * 60)
    
    # Initialize runner
    runner = PylintRunner(args.source_dir, args.config)
    
    # Apply fixes first if requested
    if args.fix:
        print("🔧 Applying automatic fixes...")
        runner.apply_autopep8_fixes()
        runner.apply_black_formatting()
        print()
    
    # Run analysis
    results = runner.run_pylint_analysis(args.format)
    
    if args.format == "json":
        # Analyze and display results
        runner.analyze_results()
        
        # Generate detailed report
        runner.generate_report(args.report)
    
    print("\n🎉 Pylint analysis complete!")
    
    # Return appropriate exit code
    if isinstance(results, list) and results:
        # Issues found
        sys.exit(1)
    else:
        # No issues or successful run
        sys.exit(0)


if __name__ == "__main__":
    main()
