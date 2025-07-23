# Kill any process running on port 8000 (Windows PowerShell)
$port = 8000
Write-Host "Checking if server is running on port $port..."
$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
    foreach ($procId in $pids) {
        Write-Host "Found process on port $port, killing PID $procId..."
        Stop-Process -Id $procId -Force
    }
    Start-Sleep -Seconds 1
    Write-Host "Process killed."
} else {
    Write-Host "No process found on port $port."
}
