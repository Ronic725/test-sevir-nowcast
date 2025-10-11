"""
Centralized path configuration for the project.
Loads paths from config/paths.yaml and provides easy access throughout the codebase.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠️  PyYAML not available - using fallback hardcoded paths")


class ProjectPaths:
    """
    Centralized path manager for the project.
    Auto-detects project root and provides access to all configured paths.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize paths from config file or use defaults.
        
        Args:
            config_path: Path to paths.yaml config file. If None, auto-detects.
        """
        self._project_root = self._find_project_root()
        self._config: Dict[str, Any] = {}
        
        if config_path is None:
            config_path = self._project_root / "config" / "paths.yaml"
        
        if HAS_YAML and config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)  # type: ignore
        else:
            self._config = self._get_fallback_config()
    
    @staticmethod
    def _find_project_root() -> Path:
        """
        Find project root by looking for marker files/directories.
        Searches upward from current working directory.
        """
        # Try from cwd first
        cwd = Path.cwd()
        if (cwd / "config").exists() or (cwd / "src").exists():
            return cwd
        
        # Search upward for markers
        current = cwd
        for _ in range(5):  # Max 5 levels up
            if (current / "config").exists() or (current / "src").exists():
                return current
            current = current.parent
        
        # Fallback to cwd
        return cwd
    
    @staticmethod
    def _get_fallback_config() -> Dict[str, Any]:
        """Fallback configuration when YAML not available."""
        return {
            'project': {'root': '.', 'src': 'src', 'data': 'data', 'models': 'models'},
            'data': {
                'sevir': {
                    'vil_2019_h1': 'data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5',
                },
            },
            'custom': {
                'fine_tuning': {
                    'events_cache': 'custom/fine_tuning/cache/finetune_events.npz',
                    'lora_adapter': 'custom/fine_tuning/outputs/lora_adapter.npz',
                },
                'results': {
                    'model_performance': 'custom/results/model_performance',
                },
            },
            'sevir': {'mean': 33.44, 'scale': 47.54},
        }
    
    def _resolve_path(self, path_str: str) -> Path:
        """Convert relative path string to absolute Path object."""
        if path_str == ".":
            return self._project_root
        return self._project_root / path_str
    
    def get(self, key_path: str, as_string: bool = False) -> Any:
        """
        Get a configured value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'data.sevir.vil_2019_h1')
            as_string: If True, return Path as string instead of Path object
            
        Returns:
            The configured value (Path object for paths, raw value otherwise)
            
        Examples:
            >>> paths = ProjectPaths()
            >>> paths.get('data.sevir.vil_2019_h1')
            PosixPath('/Users/.../data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5')
            >>> paths.get('sevir.mean')
            33.44
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                raise KeyError(f"Invalid key path: {key_path}")
        
        # Convert string paths to Path objects
        if isinstance(value, str) and ('/' in value or value == '.'):
            path = self._resolve_path(value)
            return str(path) if as_string else path
        
        return value
    
    # Convenience properties for commonly used paths
    
    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._project_root
    
    @property
    def src(self) -> Path:
        """Source code directory."""
        return self.get('project.src')
    
    @property
    def data(self) -> Path:
        """Data directory."""
        return self.get('project.data')
    
    @property
    def models(self) -> Path:
        """Models directory."""
        return self.get('project.models')
    
    @property
    def logs(self) -> Path:
        """Logs directory."""
        return self.get('project.logs')
    
    # Data paths
    
    @property
    def sevir_vil_file(self) -> Path:
        """Main SEVIR VIL H5 file."""
        return self.get('data.sevir.vil_2019_h1')
    
    @property
    def catalog_file(self) -> Path:
        """SEVIR catalog CSV file."""
        return self.get('data.sevir.catalog')
    
    # Fine-tuning paths
    
    @property
    def finetune_cache(self) -> Path:
        """Fine-tuning events cache file."""
        return self.get('custom.fine_tuning.events_cache')
    
    @property
    def lora_adapter(self) -> Path:
        """LoRA adapter weights file."""
        return self.get('custom.fine_tuning.lora_adapter')
    
    @property
    def results_dir(self) -> Path:
        """Model performance results directory."""
        return self.get('custom.results.model_performance')
    
    # SEVIR constants
    
    @property
    def sevir_mean(self) -> float:
        """SEVIR normalization mean."""
        return self.get('sevir.mean')
    
    @property
    def sevir_scale(self) -> float:
        """SEVIR normalization scale."""
        return self.get('sevir.scale')
    
    def setup_python_path(self):
        """Add project directories to Python path."""
        sys.path.insert(0, str(self.root))
        sys.path.insert(0, str(self.src))


# Global instance for easy import
_paths_instance: Optional[ProjectPaths] = None


def get_paths() -> ProjectPaths:
    """Get or create global ProjectPaths instance."""
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = ProjectPaths()
    return _paths_instance


# Convenience function for quick access
def get_path(key: str, as_string: bool = False) -> Any:
    """
    Get a path using dot notation (convenience wrapper).
    
    Examples:
        >>> from config.project_paths import get_path
        >>> data_dir = get_path('project.data')
        >>> sevir_file = get_path('data.sevir.vil_2019_h1')
    """
    return get_paths().get(key, as_string=as_string)


if __name__ == "__main__":
    # Test the configuration
    paths = ProjectPaths()
    
    print("=" * 60)
    print("Project Path Configuration")
    print("=" * 60)
    print(f"Project Root: {paths.root}")
    print(f"Source Dir:   {paths.src}")
    print(f"Data Dir:     {paths.data}")
    print(f"Models Dir:   {paths.models}")
    print()
    print("SEVIR Paths:")
    print(f"  VIL File:   {paths.sevir_vil_file}")
    print(f"  Catalog:    {paths.catalog_file}")
    print()
    print("Fine-tuning Paths:")
    print(f"  Cache:      {paths.finetune_cache}")
    print(f"  Adapter:    {paths.lora_adapter}")
    print()
    print("SEVIR Constants:")
    print(f"  Mean:       {paths.sevir_mean}")
    print(f"  Scale:      {paths.sevir_scale}")
    print("=" * 60)
