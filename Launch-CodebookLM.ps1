<#
.SYNOPSIS
    CodebookLM v1.0 Launcher
.DESCRIPTION
    Offline AI Codebase Assistant
    Understand Any Codebase. Instantly.
    Version: 1.0
#>

$Host.UI.RawUI.WindowTitle = "CodebookLM v1.0"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "📘 CodebookLM v1.0" -ForegroundColor Cyan
Write-Host "Offline AI Codebase Assistant"
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Understand Any Codebase. Instantly.`n"

# Check for Python
if (!(Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Pause
    exit
}

Write-Host "Launching local server..." -ForegroundColor Green
Start-Process "http://localhost:8501"

# Run Streamlit
python -m streamlit run app.py --server.port 8501 --server.headless true
