param(
  [int]$Port = 8775,
  [string]$Route = "",
  [switch]$Streamlit,
  [switch]$SkipOllamaCheck,
  [switch]$NoBrowser,
  [switch]$TabMode,
  [switch]$StopServer,
  [switch]$InstallDesktopShortcut
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$Server = Join-Path $Root "backend_server.py"
$Dashboard = Join-Path $Root "dashboard.py"
$DataDir = Join-Path $Root "eval_runs"
$PidFile = Join-Path $DataDir "server.pid"
$LogFile = Join-Path $DataDir "one_tap.log"
$ServerOutLog = Join-Path $DataDir "server.log"
$ServerErrLog = Join-Path $DataDir "server.err.log"
$IconFile = Join-Path $Root "web\work121.ico"
$NativeBin = Join-Path $Root "bin\work121.exe"
$CargoManifest = Join-Path $Root "native\Cargo.toml"

if (-not (Test-Path $Server)) {
  throw "work121 backend_server.py not found at: $Server"
}

if (-not (Test-Path $DataDir)) {
  New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

function Write-TapLog {
  param([string]$Message, [string]$Color = "Gray")
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  Add-Content -Path $LogFile -Value "[$stamp] $Message" -Encoding UTF8
  Write-Host $Message -ForegroundColor $Color
}

function Install-Work121Shortcuts {
  $shell = New-Object -ComObject WScript.Shell
  $desktopPaths = New-Object System.Collections.Generic.List[string]
  $primaryDesktop = [Environment]::GetFolderPath("Desktop")
  if ($primaryDesktop -and (Test-Path $primaryDesktop)) {
    $desktopPaths.Add($primaryDesktop)
  }
  $localDesktop = Join-Path $env:USERPROFILE "Desktop"
  if ((Test-Path $localDesktop) -and ($localDesktop -notin $desktopPaths)) {
    $desktopPaths.Add($localDesktop)
  }

  $thisScript = Join-Path $Root "one_tap.ps1"
  foreach ($desk in $desktopPaths) {
    $lnkPath = Join-Path $desk "work121 Workbench.lnk"
    $sc = $shell.CreateShortcut($lnkPath)
    $sc.TargetPath = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    $sc.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$thisScript`" -Port $Port"
    $sc.WorkingDirectory = $Root
    if (Test-Path $IconFile) {
      $sc.IconLocation = "$IconFile,0"
    }
    $sc.WindowStyle = 7
    $sc.Description = "work121 - Wire, Interoperability and LLM Boundary Workbench (One-Tap Launch)"
    $sc.Save()
    Write-TapLog "Installed Desktop shortcut: $lnkPath" "Green"
  }
}

if ($InstallDesktopShortcut) {
  Install-Work121Shortcuts
  if ($NoBrowser) {
    exit 0
  }
}

if ($StopServer) {
  $stoppedAny = $false
  if (Test-Path $PidFile) {
    $rawPid = (Get-Content $PidFile -Raw).Trim()
    if ($rawPid -match '^\d+$') {
      Stop-Process -Id ([int]$rawPid) -Force -ErrorAction SilentlyContinue
      Write-TapLog "Stopped work121 server (PID $rawPid)." "Yellow"
      $stoppedAny = $true
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  }
  $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
  foreach ($ownerPid in $listeners) {
    if ($ownerPid -and $ownerPid -gt 4) {
      Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
      Write-TapLog "Stopped process listening on port $Port (PID $ownerPid)." "Yellow"
      $stoppedAny = $true
    }
  }
  if (-not $stoppedAny) {
    Write-TapLog "No active work121 server found on port $Port." "DarkGray"
  }
  exit 0
}

# 1. Ensure native Rust binary gate (bin\work121.exe) is built
if (-not (Test-Path $NativeBin) -and (Test-Path $CargoManifest)) {
  $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
  if ($cargoCmd) {
    try {
      Write-TapLog "Building native Rust wire boundary gate (native\Cargo.toml)..." "Cyan"
      & $cargoCmd.Source build --release --manifest-path $CargoManifest | Out-Null
      $builtExe = Join-Path $Root "native\target\release\work121.exe"
      if (Test-Path $builtExe) {
        New-Item -ItemType Directory -Force (Join-Path $Root "bin") | Out-Null
        Copy-Item -Force $builtExe $NativeBin
        Write-TapLog "Compiled native binary: $NativeBin" "Green"
      }
    } catch {
      Write-TapLog ("Native binary build warning: " + $_.Exception.Message) "Yellow"
    }
  }
}

# 2. Ensure local Ollama daemon is running if installed
if (-not $SkipOllamaCheck) {
  $tagsUrl = "http://127.0.0.1:11434/api/tags"
  $ollamaUp = $false
  try {
    $null = Invoke-RestMethod -Uri $tagsUrl -TimeoutSec 2
    $ollamaUp = $true
  } catch {}

  if (-not $ollamaUp) {
    $ollamaExe = $null
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { $ollamaExe = $cmd.Source }
    if (-not $ollamaExe) {
      $fallbackExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
      if (Test-Path $fallbackExe) { $ollamaExe = $fallbackExe }
    }
    if ($ollamaExe) {
      Write-TapLog "Starting local Ollama daemon: $ollamaExe serve" "Cyan"
      Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
      for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 400
        try {
          $null = Invoke-RestMethod -Uri $tagsUrl -TimeoutSec 2
          $ollamaUp = $true
          break
        } catch {}
      }
    }
  }
  if ($ollamaUp) {
    Write-TapLog "Local Ollama endpoint ready at http://127.0.0.1:11434" "Green"
  }
}

# 3. Ensure unified work121 backend & Web UI server is running on $Port
$BaseUrl = "http://127.0.0.1:$Port"
$HealthUrl = "$BaseUrl/health"
$serverUp = $false

try {
  $hz = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
  if ($hz.status -eq "ok" -and $hz.server -eq "work121-unified-workbench") {
    $serverUp = $true
    Write-TapLog "work121 Unified Workbench already live at $BaseUrl" "Green"
  }
} catch {}

$pyExe = $null
foreach ($venvCand in @((Join-Path $Root ".venv\Scripts\python.exe"), (Join-Path $Root ".venv\Scripts\pythonw.exe"))) {
  if (Test-Path -LiteralPath $venvCand -PathType Leaf) {
    $pyExe = $venvCand
    break
  }
}
if (-not $pyExe) {
  $foundPy = Get-Command python -ErrorAction SilentlyContinue
  if ($foundPy) { $pyExe = $foundPy.Source }
}
if (-not $pyExe) {
  throw "Python 3 not found. Ensure .venv or system Python is installed."
}

if (-not $serverUp) {
  Write-TapLog "Launching work121 Unified Workbench server at $BaseUrl..." "Cyan"
  $argStr = "`"$Server`" --host 127.0.0.1 --port $Port"
  $proc = Start-Process -FilePath $pyExe -ArgumentList $argStr -WorkingDirectory $Root `
    -RedirectStandardOutput $ServerOutLog -RedirectStandardError $ServerErrLog `
    -WindowStyle Hidden -PassThru
  Set-Content -Path $PidFile -Value $proc.Id -Encoding ASCII

  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 350
    try {
      $hz = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
      if ($hz.status -eq "ok") {
        $serverUp = $true
        break
      }
    } catch {}
    if ($proc.HasExited -and $proc.ExitCode -ne 0) {
      break
    }
  }

  if (-not $serverUp) {
    throw "work121 server failed to become healthy at $HealthUrl. Check $ServerErrLog"
  }
  Write-TapLog "work121 Unified Workbench ready (PID $($proc.Id)): $BaseUrl/" "Green"
}

# Optional: Also launch 7-tab Streamlit workbench if -Streamlit flag passed
if ($Streamlit -and (Test-Path $Dashboard)) {
  $stArgs = "-m streamlit run `"$Dashboard`" --server.port 8501 --server.headless true"
  Start-Process -FilePath $pyExe -ArgumentList $stArgs -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 2
  $BaseUrl = "http://127.0.0.1:8501"
}

# 4. Open One-Tap App Window (Edge/Chrome --app mode preferred) or default browser
$LaunchUrl = "$BaseUrl/"
if ($Route -and -not $Streamlit) {
  $cleanRoute = $Route.TrimStart('#', '/')
  $LaunchUrl = "$BaseUrl/#/$cleanRoute"
}

if (-not $NoBrowser) {
  $openedApp = $false
  if (-not $TabMode) {
    $browserCandidates = @(
      (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
      (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
      (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
      (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"),
      (Join-Path $env:LocalAppData "Google\Chrome\Application\chrome.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($exe in $browserCandidates) {
      try {
        Start-Process -FilePath $exe -ArgumentList "--app=$LaunchUrl" | Out-Null
        $openedApp = $true
        Write-TapLog "Opened work121 app window via $exe ($LaunchUrl)" "Green"
        break
      } catch {}
    }
  }
  if (-not $openedApp) {
    Start-Process "$LaunchUrl" | Out-Null
    Write-TapLog "Opened default browser at $LaunchUrl" "Green"
  }
}
