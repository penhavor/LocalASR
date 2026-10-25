# =====================================================================
#  LocalASR - portable Python downloader (called by RUN.bat on first run).
#  Downloads the pinned full CPython build (the official Python Software
#  Foundation package on NuGet), verifies its SHA256 and unpacks it into
#  runtime\python. Console messages follow the system UI language (ru/en).
#  Exit codes: 0 = ok, 1 = failure.
# =====================================================================
$ErrorActionPreference = 'Stop'

$PythonVersion  = '3.12.10'
$Sha256Expected = '0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8'
$FileName       = "python-$PythonVersion-full.zip"
$Url            = "https://api.nuget.org/v3-flatcontainer/python/$PythonVersion/python.$PythonVersion.nupkg"

$IsRu = [System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName -eq 'ru'
function Say([string]$en, [string]$ru) {
    if ($IsRu) { Write-Host $ru } else { Write-Host $en }
}

$Root       = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $Root 'runtime'
$DestDir    = Join-Path $RuntimeDir 'python'
$ZipPath    = Join-Path $RuntimeDir $FileName
$UnpackDir  = Join-Path $RuntimeDir '_unpack'

if (Test-Path (Join-Path $DestDir 'python.exe')) {
    Say 'LocalASR: the portable Python is already in place.' 'LocalASR: портативный Python уже установлен.'
    exit 0
}

try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }
$ProgressPreference = 'SilentlyContinue'

try {
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

    # a previous attempt may have left leftovers - start clean
    if (Test-Path $DestDir)   { Remove-Item -Recurse -Force $DestDir }
    if (Test-Path $UnpackDir) { Remove-Item -Recurse -Force $UnpackDir }

    Say "LocalASR: downloading the full Python $PythonVersion (~15 MB)..." `
        "LocalASR: скачиваем полный Python $PythonVersion (~15 МБ)..."
    foreach ($attempt in 1..2) {
        try {
            $wc = New-Object System.Net.WebClient
            try { $wc.DownloadFile($Url, $ZipPath) } finally { $wc.Dispose() }
            break
        } catch {
            Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
            if ($attempt -eq 2) { throw }
            Say 'LocalASR: download failed, retrying...' 'LocalASR: не удалось скачать, пробуем ещё раз...'
            Start-Sleep -Seconds 3
        }
    }

    Say 'LocalASR: verifying the checksum...' 'LocalASR: проверяем контрольную сумму...'
    $hash = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash
    if ($hash -ne $Sha256Expected) {
        Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
        Say 'LocalASR: checksum mismatch - the file is corrupted. Run RUN.bat again.' `
            'LocalASR: контрольная сумма не совпала - файл повреждён. Запустите RUN.bat ещё раз.'
        exit 1
    }

    Say 'LocalASR: unpacking into runtime\python...' 'LocalASR: распаковываем в runtime\python...'
    Expand-Archive -Path $ZipPath -DestinationPath $UnpackDir -Force
    Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue

    # inside the package, the Python itself lives in the 'tools' subfolder
    Move-Item -Path (Join-Path $UnpackDir 'tools') -Destination $DestDir
    Remove-Item -Recurse -Force $UnpackDir -ErrorAction SilentlyContinue

    if (-not (Test-Path (Join-Path $DestDir 'python.exe'))) {
        Say 'LocalASR: python.exe is missing after unpacking. Run RUN.bat again.' `
            'LocalASR: после распаковки нет python.exe. Запустите RUN.bat ещё раз.'
        exit 1
    }

    Say 'LocalASR: the portable Python is ready.' 'LocalASR: портативный Python готов.'
    exit 0
} catch {
    Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $UnpackDir -ErrorAction SilentlyContinue
    Say "LocalASR: could not download the portable Python: $($_.Exception.Message)" `
        "LocalASR: не удалось скачать портативный Python: $($_.Exception.Message)"
    Say 'Check the internet connection and run RUN.bat again.' `
        'Проверьте подключение к интернету и запустите RUN.bat ещё раз.'
    exit 1
}
