$ErrorActionPreference = "Stop"

$requiredVariables = @(
    "RAG_DB_HOST",
    "RAG_DB_PORT",
    "RAG_DB_NAME",
    "RAG_DB_USER",
    "RAG_DB_PASSWORD"
)

foreach ($name in $requiredVariables) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        throw "缺少环境变量: $name"
    }
}

if (-not $env:RAG_MODEL_PATH) {
    $localModel = Join-Path $PSScriptRoot ".cache\models\bge-m3"
    $env:RAG_MODEL_PATH = if (Test-Path (Join-Path $localModel "pytorch_model.bin")) {
        $localModel
    } else {
        "BAAI/bge-m3"
    }
}
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
}

& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn api:app --app-dir "$PSScriptRoot\rag_newer" --host 0.0.0.0 --port 8000
