# Stop on error
$ErrorActionPreference = "Stop"

Write-Host "🔹 Creating Python 3.13 virtual environment..."
python3.13 -m venv venv

Write-Host "🔹 Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Host "🔹 Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "🔹 Installing requirements.txt..."
pip install -r requirements.txt

Write-Host "🔹 Detecting NVIDIA GPU..."

$installCpu = $true

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $vramMB = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
        Select-Object -First 1

    $vramGB = [math]::Floor([int]$vramMB / 1024)

    Write-Host "✔ NVIDIA GPU detected with $vramGB GB VRAM"

    if ($vramGB -ge 4) {
        $installCpu = $false
    }
} else {
    Write-Host "✖ No NVIDIA GPU detected"
}

Write-Host "🔹 Installing PyTorch..."

if ($installCpu) {
    Write-Host "➡ Installing CPU-only PyTorch"
    pip install torch torchvision torchaudio
} else {
    Write-Host "➡ Installing CUDA-enabled PyTorch"
    # CUDA 13 wheels (recommended default)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
}

Write-Host "🔹 Running main.py..."
python main.py