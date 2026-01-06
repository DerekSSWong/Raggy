#Initialising variables
$VenvPath = ".venv"

Write-Host "Activating virtual environment..."
& "$VenvPath\Scripts\Activate.ps1"

Write-Host "Running program..."
python main.py