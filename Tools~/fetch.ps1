# Downloads the tinybvh test scenes used by the C# port's tests, as of tinybvh's 1.8.0 release.
# Usage: fetch.ps1 [-Dest <folder>]. The folder defaults to the TestData folder of the Unity
# project the package is embedded in, three levels above this script's folder.
param(
	[string] $Dest = ""
)
$base = "https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/testdata"
if ($Dest -eq "")
{
	$project = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
	$Dest = Join-Path $project "TestData"
}
if (-not (Test-Path $Dest))
{
	New-Item -ItemType Directory -Path $Dest | Out-Null
}
foreach ($name in @("bunny.bin", "suzanne.bin", "cryteksponza.bin"))
{
	$out = Join-Path $Dest $name
	if (-not (Test-Path $out))
	{
		Write-Host "Fetching $name"
		Invoke-WebRequest -Uri "$base/$name" -OutFile $out
	}
}
