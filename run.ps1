param([ValidateSet('prepare','train','dashboard','test','all')][string]$Action = 'dashboard')
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($Action -eq 'prepare' -or $Action -eq 'all') {
    & $projectPython -m src.prepare
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Action -eq 'train' -or $Action -eq 'all') {
    & $projectPython -m src.train
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Action -eq 'dashboard') {
    & $projectPython -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
}
if ($Action -eq 'test') { & $projectPython -m pytest -q }
exit $LASTEXITCODE
