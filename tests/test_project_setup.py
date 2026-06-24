"""
测试项目初始化和依赖安装
遵循TDD原则：先写测试，看失败，再实现
"""
import os
import pytest
import sys


class TestDirectoryStructure:
    """测试项目目录结构"""

    def test_project_root_exists(self):
        """测试项目根目录存在"""
        project_root = "/Users/marhozen/Desktop/claude/dna-ssl-project"
        assert os.path.exists(project_root), f"项目根目录不存在: {project_root}"

    def test_data_directory_exists(self):
        """测试data目录存在"""
        data_dir = "/Users/marhozen/Desktop/claude/dna-ssl-project/data"
        assert os.path.exists(data_dir), "data目录不存在"
        assert os.path.isdir(data_dir), "data路径不是目录"

    def test_src_directory_exists(self):
        """测试src目录存在"""
        src_dir = "/Users/marhozen/Desktop/claude/dna-ssl-project/src"
        assert os.path.exists(src_dir), "src目录不存在"
        assert os.path.isdir(src_dir), "src路径不是目录"

    def test_tests_directory_exists(self):
        """测试tests目录存在"""
        tests_dir = "/Users/marhozen/Desktop/claude/dna-ssl-project/tests"
        assert os.path.exists(tests_dir), "tests目录不存在"
        assert os.path.isdir(tests_dir), "tests路径不是目录"

    def test_docs_directory_exists(self):
        """测试docs目录存在"""
        docs_dir = "/Users/marhozen/Desktop/claude/dna-ssl-project/docs"
        assert os.path.exists(docs_dir), "docs目录不存在"
        assert os.path.isdir(docs_dir), "docs路径不是目录"

    def test_requirements_file_exists(self):
        """测试requirements.txt文件存在"""
        req_file = "/Users/marhozen/Desktop/claude/dna-ssl-project/requirements.txt"
        assert os.path.exists(req_file), "requirements.txt文件不存在"
        assert os.path.isfile(req_file), "requirements.txt路径不是文件"


class TestDependencies:
    """测试依赖包可导入"""

    def test_torch_importable(self):
        """测试PyTorch可导入"""
        import torch
        assert torch.__version__ is not None

    def test_numpy_importable(self):
        """测试NumPy可导入"""
        import numpy as np
        assert np.__version__ is not None

    def test_pandas_importable(self):
        """测试Pandas可导入"""
        import pandas as pd
        assert pd.__version__ is not None

    def test_sklearn_importable(self):
        """测试scikit-learn可导入"""
        import sklearn
        assert sklearn.__version__ is not None

    def test_matplotlib_importable(self):
        """测试Matplotlib可导入"""
        import matplotlib
        assert matplotlib.__version__ is not None

    def test_biopython_importable(self):
        """测试BioPython可导入"""
        import Bio
        assert Bio.__version__ is not None


class TestConfigurationFiles:
    """测试配置文件"""

    def test_config_directory_exists(self):
        """测试配置目录存在"""
        config_dir = "/Users/marhozen/Desktop/claude/dna-ssl-project/config"
        assert os.path.exists(config_dir), "config目录不存在"
        assert os.path.isdir(config_dir), "config路径不是目录"

    def test_main_config_exists(self):
        """测试主配置文件存在"""
        config_file = "/Users/marhozen/Desktop/claude/dna-ssl-project/config/default.yaml"
        assert os.path.exists(config_file), "default.yaml配置文件不存在"
        assert os.path.isfile(config_file), "default.yaml路径不是文件"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
