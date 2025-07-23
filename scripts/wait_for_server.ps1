Write-Host "Waiting for FastAPI server to start..."
for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "Server is ready!"
            exit 0
        }
    } catch {}
    Write-Host "Attempt ${i}: Server not ready, waiting..."
    Start-Sleep -Seconds 2
}
Write-Host "Server failed to start within 120 seconds"
exit 1
