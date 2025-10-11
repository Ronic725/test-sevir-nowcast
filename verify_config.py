#!/usr/bin/env python3
"""
Comprehensive verification that centralized configuration is working across all modules
"""

import sys
from pathlib import Path
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_config_import():
    """Test that config can be imported"""
    print("=" * 60)
    print("Testing Config Import")
    print("=" * 60)
    try:
        from config.project_paths import get_paths
        paths = get_paths()
        print("✅ Config imported successfully")
        print(f"   Project root: {paths.root}")
        print(f"   Data dir: {paths.data}")
        print(f"   Models dir: {paths.models}")
        print(f"   SEVIR mean: {paths.sevir_mean}")
        print(f"   SEVIR scale: {paths.sevir_scale}")
        return True, paths
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_module_imports():
    """Test that updated modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules = [
        ("Fine-tuning (PEFT)", "custom.fine_tuning.quick_peft_finetune"),
        ("Fine-tuning (Select Events)", "custom.fine_tuning.select_events_v2"),
        ("Streaming Demo", "custom.streaming.run_realtime_demo"),
        ("Streaming Predictor", "custom.streaming.realtime_predictor"),
        ("Testing (Actual SEVIR)", "custom.testing.test_actual_sevir"),
        ("Testing (Compare Models)", "custom.testing.compare_models"),
        ("Training (Limited Data)", "custom.training.train_limited_data"),
        ("Training (Quick Start)", "custom.training.quick_start_training"),
        ("Experiments (Download)", "custom.experiments.download_sevir_data"),
        ("SEVIR Dataset (Inspect)", "custom.sevir_dataset.inspect_sevir_data_refactored"),
    ]
    
    success_count = 0
    failed = []
    
    for name, module_path in modules:
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            print(f"✅ {name:30s} - {module_path}")
            success_count += 1
        except Exception as e:
            error_msg = str(e).split('\n')[0][:60]
            print(f"❌ {name:30s} - {module_path}")
            print(f"   Error: {error_msg}...")
            failed.append((name, module_path, str(e)))
    
    print(f"\n📊 {success_count}/{len(modules)} modules imported successfully")
    
    if failed and len(failed) <= 3:
        print("\n⚠️  Failed imports details:")
        for name, path, error in failed:
            print(f"\n   {name} ({path}):")
            print(f"   {error[:200]}")
    
    return success_count == len(modules)

def test_path_access():
    """Test that paths can be accessed from modules"""
    print("\n" + "=" * 60)
    print("Testing Path Access")
    print("=" * 60)
    
    try:
        from config.project_paths import get_paths
        paths = get_paths()
        
        tests = [
            ("SEVIR VIL file", paths.sevir_vil_file, True, "Required for real data"),
            ("Catalog file", paths.catalog_file, True, "Required for data loading"),
            ("Models directory", paths.models, False, "Created on first use"),
            ("Data directory", paths.data, False, "Should exist"),
            ("Fine-tune cache", paths.finetune_cache, True, "Required for PEFT"),
            ("Results directory", paths.results_dir, False, "Created on first use"),
            ("Logs directory", paths.logs, False, "Created on first use"),
        ]
        
        all_critical_exist = True
        
        for name, path, is_critical, description in tests:
            if path is None:
                status = "⚠️ " if is_critical else "📝"
                print(f"{status} {name:25s} - None ({description})")
                if is_critical:
                    all_critical_exist = False
                continue
                
            exists = path.exists()
            if is_critical:
                status = "✅" if exists else "❌"
                exists_msg = "exists" if exists else "NOT FOUND"
                if not exists:
                    all_critical_exist = False
            else:
                status = "✅" if exists else "📁"
                exists_msg = "exists" if exists else "will be created"
            
            print(f"{status} {name:25s} ({exists_msg})")
            print(f"   {path}")
        
        return all_critical_exist
    except Exception as e:
        print(f"❌ Failed to access paths: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_usage():
    """Test that scripts correctly import and use config"""
    print("\n" + "=" * 60)
    print("Testing Config Usage in Scripts")
    print("=" * 60)
    
    scripts_to_check = { # only test part of the scripts to avoid long output
        "custom/fine_tuning/quick_peft_finetune.py": ["from config.project_paths import get_paths", "paths = get_paths()"],
        "custom/streaming/run_realtime_demo.py": ["from config.project_paths import get_paths", "paths.sevir_vil_file"],
        "custom/testing/test_actual_sevir.py": ["from config.project_paths import get_paths", "paths.sevir_mean"],
        "custom/training/train_limited_data.py": ["from config.project_paths import get_paths", "paths = get_paths()"],
        "custom/streaming/realtime_predictor.py": ["from config.project_paths import get_paths", "paths.sevir_mean"],
        "custom/streaming/sevir_data_streamer.py": ["from config.project_paths import get_paths", "paths.sevir_mean"],
    }
    
    all_correct = True
    checked = 0
    
    for script_path, required_patterns in scripts_to_check.items():
        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            print(f"⚠️  {script_path} - File not found")
            continue
        
        try:
            content = full_path.read_text()
            missing_patterns = [p for p in required_patterns if p not in content]
            
            if missing_patterns:
                print(f"❌ {script_path}")
                print(f"   Missing: {', '.join(missing_patterns[:2])}")
                all_correct = False
            else:
                print(f"✅ {script_path}")
                checked += 1
        except Exception as e:
            print(f"❌ {script_path} - Error: {e}")
            all_correct = False
    
    print(f"\n📊 {checked}/{len(scripts_to_check)} scripts verified")
    return all_correct

def test_constants_usage():
    """Test that SEVIR constants are correctly used from config"""
    print("\n" + "=" * 60)
    print("Testing SEVIR Constants Usage")
    print("=" * 60)
    
    from config.project_paths import get_paths
    paths = get_paths()
    
    # Files that should use constants from config (core files only)
    files_to_check = [
        ("custom/streaming/realtime_predictor.py", True),  # Must use constants
        ("custom/streaming/sevir_data_streamer.py", True),  # Must use constants
        ("custom/testing/test_actual_sevir.py", True),  # Must use constants
        ("custom/streaming/run_realtime_demo.py", False),  # Coordinator, doesn't need constants directly
    ]
    
    critical_issues = []
    correct = 0
    total_critical = sum(1 for _, is_critical in files_to_check if is_critical)
    
    for file_path, is_critical in files_to_check:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            continue
            
        try:
            content = full_path.read_text()
            
            # Check if using config constants
            uses_config_mean = "paths.sevir_mean" in content or "SEVIR_MEAN = paths.sevir_mean" in content
            uses_config_scale = "paths.sevir_scale" in content or "SEVIR_SCALE = paths.sevir_scale" in content
            
            if uses_config_mean and uses_config_scale:
                print(f"✅ {file_path}")
                correct += 1
            elif is_critical:
                print(f"❌ {file_path} - Missing config constants (CRITICAL)")
                critical_issues.append(file_path)
            else:
                print(f"📝 {file_path} - Doesn't use constants directly (OK)")
        except Exception as e:
            print(f"❌ {file_path} - Error: {e}")
            if is_critical:
                critical_issues.append(file_path)
    
    print(f"\n📊 {correct}/{total_critical} critical files use config constants")
    return len(critical_issues) == 0

def main():
    print("\n🔍 Comprehensive Centralized Configuration Verification")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {PROJECT_ROOT}")
    print()
    
    results = []
    
    # Run tests in order
    config_success, paths = test_config_import()
    results.append(("Config Import", config_success))
    
    if config_success:
        results.append(("Config Usage", test_config_usage()))
        results.append(("Constants Usage", test_constants_usage()))
        results.append(("Path Access", test_path_access()))
        results.append(("Module Imports", test_module_imports()))
    else:
        print("\n⚠️  Skipping subsequent tests due to config import failure")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(success for _, success in results)
    passed_count = sum(1 for _, success in results if success)
    
    print(f"\n📈 Results: {passed_count}/{len(results)} tests passed")
    
    if all_passed:
        print("\n🎉 All tests passed! Configuration is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run scripts from workspace root:")
        print("      python3 custom/streaming/run_realtime_demo.py --help")
        print("      python3 custom/testing/test_actual_sevir.py --help")
        print("      python3 custom/training/quick_start_training.py")
        print("   2. Check QUICK_REFERENCE.md for usage examples")
        print("   3. Test individual scripts for end-to-end functionality")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
        failed_tests = [name for name, success in results if not success]
        print("\n❌ Failed tests:")
        for test_name in failed_tests:
            print(f"   • {test_name}")
        print("\n📖 Troubleshooting:")
        print("   1. Ensure config/paths.yaml exists and is properly formatted")
        print("   2. Check that all scripts import: 'from config.project_paths import get_paths'")
        print("   3. Verify SEVIR data files are in the correct locations")
        print("   4. Review QUICK_REFERENCE.md for migration guidelines")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
