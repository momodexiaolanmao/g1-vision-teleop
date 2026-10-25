param([switch]$UseTsinghuaMirror)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
if ($UseTsinghuaMirror) {
    & '.\.venv\Scripts\python.exe' -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
} else {
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (Test-Path -LiteralPath 'unitree_g1_walking_mapping.json') {
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements-walking.txt
    if ($LASTEXITCODE -ne 0) { throw 'G1 walking dependency installation failed.' }
}
& '.\.venv\Scripts\python.exe' download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Pose model download failed.' }
if (Test-Path -LiteralPath 'unitree_g1_walking_mapping.json') {
    & '.\.venv\Scripts\python.exe' download_model.py --variant full
    if ($LASTEXITCODE -ne 0) { throw 'Full pose model download failed.' }
}
Write-Host 'Ready. Run: .\.venv\Scripts\python.exe teleop.py --demo'
