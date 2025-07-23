Write-Host "Waiting extra time for FastAPI server to start in compound mode..."
Start-Sleep -Seconds 5
for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "Server is ready!"
            exit 0
        }
    } catch {}
    Write-Host "Attempt ${i}: Server not ready, waiting..."
    Start-Sleep -Seconds 3
}
Write-Host "Server failed to start within 180 seconds"
exit 1
