$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Sdk = Resolve-Path "$Root\..\tools\android-sdk"
$BuildTools = "$Sdk\build-tools\35.0.0"
$Platform = "$Sdk\platforms\android-35\android.jar"
$FinalOut = "$Root\build"
$Stamp = [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')
$Out = "$FinalOut\work_$Stamp"
$Src = "$Root\app\src\main"
$Package = "com.smartclassroom.edge"

New-Item -ItemType Directory -Force -Path $Out, $FinalOut | Out-Null
New-Item -ItemType Directory -Force -Path "$Out\compiled", "$Out\classes", "$Out\dex" | Out-Null

& "$BuildTools\aapt2.exe" compile --dir "$Src\res" -o "$Out\compiled\res.zip"
& "$BuildTools\aapt2.exe" link `
    -I "$Platform" `
    --min-sdk-version 23 `
    --target-sdk-version 35 `
    --manifest "$Src\AndroidManifest.xml" `
    -o "$Out\base-unsigned.apk" `
    --java "$Out\generated" `
    "$Out\compiled\res.zip"

$JavaFiles = Get-ChildItem -Recurse -Filter *.java "$Src\java", "$Out\generated" | ForEach-Object { $_.FullName }
& javac -encoding UTF-8 -source 8 -target 8 -bootclasspath "$Platform" -d "$Out\classes" $JavaFiles

$ClassFiles = Get-ChildItem -Recurse -Filter *.class "$Out\classes" | ForEach-Object { $_.FullName }
& "$BuildTools\d8.bat" --min-api 23 --lib "$Platform" --output "$Out\dex" $ClassFiles

& "$BuildTools\aapt2.exe" link `
    -I "$Platform" `
    --min-sdk-version 23 `
    --target-sdk-version 35 `
    --manifest "$Src\AndroidManifest.xml" `
    -o "$Out\app-unsigned.apk" `
    --java "$Out\generated2" `
    -A "$Src\assets" `
    "$Out\compiled\res.zip"
Copy-Item "$Out\dex\classes.dex" "$Out\classes.dex" -Force
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$apk = [System.IO.Compression.ZipFile]::Open($("$Out\app-unsigned.apk"), [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $existing = $apk.GetEntry("classes.dex")
    if ($existing -ne $null) {
        $existing.Delete()
    }
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($apk, "$Out\classes.dex", "classes.dex") | Out-Null
} finally {
    $apk.Dispose()
}

$Keystore = "$FinalOut\debug.keystore"
if (-not (Test-Path $Keystore)) {
    & keytool -genkeypair -v `
        -keystore "$Keystore" `
        -storepass android `
        -keypass android `
        -alias androiddebugkey `
        -keyalg RSA `
        -keysize 2048 `
        -validity 10000 `
        -dname "CN=Android Debug,O=Smart Classroom,C=CN"
}

$SignedApk = "$FinalOut\SmartClassroomEdge_$Stamp.apk"
& "$BuildTools\zipalign.exe" -f 4 "$Out\app-unsigned.apk" "$Out\SmartClassroomEdge-aligned.apk"
& "$BuildTools\apksigner.bat" sign `
    --ks "$Keystore" `
    --ks-pass pass:android `
    --key-pass pass:android `
    --out "$SignedApk" `
    "$Out\SmartClassroomEdge-aligned.apk"
& "$BuildTools\apksigner.bat" verify "$SignedApk"
Copy-Item "$SignedApk" "$FinalOut\SmartClassroomEdge.apk" -Force

Write-Host "Built $SignedApk"
Write-Host "Copied to $FinalOut\SmartClassroomEdge.apk"
