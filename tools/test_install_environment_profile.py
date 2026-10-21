"""Installer regression tests: commands are mocked, never install or download."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import project_runtime


@pytest.fixture
def installer(monkeypatch):
    monkeypatch.setitem(sys.modules, 'project_runtime', project_runtime)
    path = Path(__file__).with_name('install_environment_profile.py')
    spec = importlib.util.spec_from_file_location('installer_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile():
    return {'external_manager': {'kind': 'uv_lock', 'repository': 'https://example.invalid/verl.git',
            'tag': 'v0.9.1', 'commit': 'fixed', 'python': '3.12', 'extras': ['vllm']},
            'isolation': {'require_user_site_disabled': True, 'require_clean_sys_path': True}}


def test_dry_run_does_not_run_commands_or_create_target(installer, profile, tmp_path, monkeypatch):
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/fake/' + name)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected subprocess execution'))
    target = tmp_path / 'new'
    assert installer.install_uv_lock_profile(root=tmp_path, profile_id='PT-VERL-CUDA', profile=profile,
        target_value=str(target), python_version='3.12', reuse=False, dry_run=True) == 0
    assert not target.exists()


@pytest.mark.parametrize('git_directory,reuse', [(True, False), (False, True)])
def test_existing_directory_guard(installer, profile, tmp_path, monkeypatch, git_directory, reuse):
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/fake/' + name)
    if git_directory:
        (tmp_path / '.git').mkdir()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Must stop before installation'))
    with pytest.raises(RuntimeError):
        installer.install_uv_lock_profile(root=tmp_path, profile_id='PT-VERL-CUDA', profile=profile,
            target_value=str(tmp_path), python_version='3.12', reuse=reuse, dry_run=False)


def test_wrong_commit_blocks_sync(installer, profile, tmp_path, monkeypatch):
    (tmp_path / '.git').mkdir()
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/fake/' + name)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='wrong\n')
    monkeypatch.setattr(subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='版本不匹配'):
        installer.install_uv_lock_profile(root=tmp_path, profile_id='PT-VERL-CUDA', profile=profile,
            target_value=str(tmp_path), python_version='3.12', reuse=True, dry_run=False)
    assert len(calls) == 1 and calls[0][1:] == ['rev-parse', 'HEAD']


def test_uv_plan_uses_pinned_python_extra_and_isolation(installer, profile, tmp_path, monkeypatch):
    (tmp_path / '.git').mkdir()
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/fake/' + name)
    monkeypatch.setenv('PYTHONPATH', '/foreign/site')
    monkeypatch.setenv('ALL_PROXY', 'socks5://example.invalid:1080')
    monkeypatch.setenv('HTTPS_PROXY', 'http://example.invalid:8080')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='fixed\n', returncode=0))
    calls = []
    monkeypatch.setattr(installer, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(installer, 'check_uv_dependencies', lambda **kwargs: None)
    installer.install_uv_lock_profile(root=tmp_path, profile_id='PT-VERL-CUDA', profile=profile,
        target_value=str(tmp_path), python_version='3.12', reuse=True, dry_run=False)
    command, options = calls[0]
    assert command[1:] == ['sync', '--frozen', '--all-packages', '--python', '3.12', '--extra', 'vllm']
    assert options['env']['PYTHONPATH'] == '' and options['env']['PYTHONNOUSERSITE'] == '1'
    assert 'ALL_PROXY' not in options['env'] and 'HTTPS_PROXY' in options['env']
    assert calls[1][0][0] == str(tmp_path / '.venv/bin/python')


@pytest.mark.parametrize('count,code,extra,allowed_ok', [(1, 1, '', True), (2, 1, '', False),
                                                      (1, 2, '', False), (2, 1, '\nThe package `bad` requires other', False)])
def test_only_exact_complete_metadata_exception_allowed(installer, tmp_path, monkeypatch, count, code, extra, allowed_ok):
    issue = 'The package `mistral-common` requires a registered NumPy range'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=code,
        stdout=issue + extra, stderr=f'Found {count} incompatibility'))
    kwargs = dict(uv='uv', python=tmp_path / 'python', profile={'allowed_pip_check_issues': [issue]},
                  cwd=tmp_path, env={}, dry_run=False)
    if allowed_ok:
        installer.check_uv_dependencies(**kwargs)
    else:
        with pytest.raises(RuntimeError):
            installer.check_uv_dependencies(**kwargs)


def test_clone_retry_is_bounded(installer, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs:
                        (calls.append(command) or SimpleNamespace(returncode=128)))
    with pytest.raises(RuntimeError, match='连续失败 3 次'):
        installer.clone_locked_repository(git='git', repository='https://example.invalid/repo',
            tag='v0.9.1', target=tmp_path / 'absent', dry_run=False)
    assert len(calls) == 3


@pytest.mark.parametrize('profile_id,arguments', [('PT-CPU', ['--target', '/unused']),
                                               ('PT-VERL-CUDA', ['--env-name', 'unused'])])
def test_cli_rejects_wrong_manager_arguments(installer, profile, tmp_path, monkeypatch, profile_id, arguments):
    monkeypatch.setattr(installer, 'bootstrap_project_root', lambda: tmp_path)
    monkeypatch.setattr(installer, 'load_environment_profile', lambda *a: profile if profile_id == 'PT-VERL-CUDA' else {})
    monkeypatch.setattr(sys, 'argv', ['installer', '--profile', profile_id, *arguments, '--dry-run'])
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 2


def test_conda_dry_run_keeps_existing_install_path(installer, tmp_path, monkeypatch):
    profile = {'requirements': ['requirements/torch-cpu.txt'],
               'constraint_files': ['requirements/post-training-cpu-constraints.txt'],
               'isolation': {'require_user_site_disabled': True, 'require_clean_sys_path': True}}
    monkeypatch.setattr(installer, 'bootstrap_project_root', lambda: tmp_path)
    monkeypatch.setattr(installer, 'load_environment_profile', lambda *a: profile)
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/fake/' + name)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Dry run must not execute commands'))
    monkeypatch.setattr(sys, 'argv', ['installer', '--profile', 'PT-CPU', '--env-name', 'cpu_test', '--dry-run'])
    calls = []
    monkeypatch.setattr(installer, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    assert installer.main() == 0
    assert calls[0][0] == ['/fake/conda', 'create', '-n', 'cpu_test', 'python=3.12', '-y']
    assert all(options['dry_run'] for _, options in calls)
    assert any('-c' in command for command, _ in calls)
    assert any('ipykernel' in command for command, _ in calls)
