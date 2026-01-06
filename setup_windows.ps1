#Initialising variables
$VenvPath = ".venv"


# Check if Python 3.13 is available via the Python Launcher
$pyVersion = py -3.13 -V 2>$null
if ($pyVersion) {
    Write-Host "Python 3.13 detected"
}
else {
    Write-Host "Python 3.13 not found. Installing..."
    py install 3.13
}

# Create virtual environment
Write-Host "Creating virtual environment..."
py -3.13 -m venv $VenvPath


# Activate the virtual environment
Write-Host "Activating virtual environment..."
& "$VenvPath\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Detect cuda version and vram with nvidia-smi, then install pytorch accordingly
$cudaVersion = & nvidia-smi | Select-String "CUDA Version" | ForEach-Object {
    ($_ -split "CUDA Version:")[1].Trim()
}

if ($cudaVersion) {
	
	#Format cuda version
	$cudaVersion = ([float]($cudaVersion -replace "[\s|]", "").Trim())
	Write-Host "CUDA Version: $cudaVersion"
	
	#Check for VRAM amount
	$vramMB = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits).Trim()
	$vramGB = [float]([math]::Round($vramMB / 1024, 2))
	Write-Host "VRAM: $vramGB GB"
	
	if ($cudaVersion -gt 13 -and $vramGB -ge 4) {
		Write-Host "Installing pyTorch for CUDA 13+"
		pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	}
	elseif ($cudaVersion -gt 12.8 -and $vramGB -ge 4){
		Write-Host "Installing pyTorch for CUDA 12.8+"
		pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	}
	elseif($cudaVersion -gt 12.6 -and $vramGB -ge 4){
		Write-Host "Installing pyTorch for CUDA 12.6+"
		pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	}
	else{
		Write-Host "VRAM requirement not met, installing pyTorch for CPU"
		pip install torch torchvision
	}
	
} else {
    Write-Host "CUDA requirement not met, installing pyTorch for CPU"
	pip install torch torchvision
}

# Install other requirements
pip install -r req.txt


Write-Host "Setup complete. Press any key to exit..."
[System.Console]::ReadKey($true)