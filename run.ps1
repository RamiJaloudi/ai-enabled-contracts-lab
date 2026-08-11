$ErrorActionPreference = "Stop"
Write-Host "AI-Enabled Contracts Lab URL: http://localhost:8501" -ForegroundColor Cyan
uv run --extra cu129 streamlit run streamlit_app.py --server.address=localhost --server.port=8501
