#!/usr/bin/env python3
"""
Verify that centralized configuration is working across all modules
"""

import sys
from pathlib import Path

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
        return True
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False

def test_module_imports():
    """Test that updated modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules = [
        ("Fine-tuning", "custom.fine_tuning.quick_peft_finetune"),
        ("Streaming Demo", "custom.streaming.run_realtime_demo"),
        ("Testing", "custom.testing.test_actual_sevir"),
        ("Training", "custom.training.train_limited_data"),
    ]
    
    success_count = 0
    for name, module_path in modules:
        try:
            # Try to import the module
            parts = module_path.split('.')
            module = __import__(module_path)
            for part in parts[1:]:
                module = getattr(module, part)
            print(f"✅ {name:20s} - {module_path}")
            success_count += 1
        except Exception as e:
            print(f"⚠️  {name:20s} - {module_path}")
            print(f"   Error: {str(e)[:50]}...")
    
    print(f"\n{success_count}/{len(modules)} modules imported successfully")
    return success_count == len(modules)

def test_path_access():
    """Test that paths can be accessed from modules"""
    print("\n" + "=" * 60)
    print("Testing Path Access in Modules")
    print("=" * 60)
    
    try:
        from config.project_paths import get_paths
        paths = get_paths()
        
        tests = [
            ("SEVIR VIL file", paths.sevir_vil_file, True),
            ("Catalog file", paths.catalog_file, True),
            ("Models directory", paths.models, False),
            ("Data directory", paths.data, False),
            ("Fine-tune cache", paths.finetune_cache, True),
            ("Results directory", paths.results_dir, False),
        ]
        
        for name, path, check_exists in tests:
            if path is None:
                print(f"⚠️  {name:25s} - None (not configured)")
                continue
            if check_exists:
                status = "✅" if path.exists() else "⚠️ "
                exists_msg = "exists" if path.exists() else "not found"
                print(f"{status} {name:25s} - {path} ({exists_msg})")
            else:
                status = "✅" if path.exists() else "📁"
                exists_msg = "exists" if path.exists() else "needs creation"
                print(f"{status} {name:25s} - {path} ({exists_msg})")
        
        return True
    except Exception as e:
        print(f"❌ Failed to access paths: {e}")
        return False

def main():
    print("\n🔍 Centralized Configuration Verification")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {PROJECT_ROOT}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Config Import", test_config_import()))
    results.append(("Module Imports", test_module_imports()))
    results.append(("Path Access", test_path_access()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Configuration is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run scripts from workspace root:")
        print("      python3 custom/streaming/run_realtime_demo.py --help")
        print("      python3 custom/testing/test_actual_sevir.py --help")
        print("      python3 custom/training/quick_start_training.py")
        print("   2. Check CENTRALIZED_CONFIG_MIGRATION.md for details")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
