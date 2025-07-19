$env:PYTHONUNBUFFERED=1
$env:PYTHONPATH = "D:\WorkSpace\progamme\Codefield\Code_Py\ccs_23_oars_stateful_attacks\attacks"

$paths = @("boundary\untargeted", "hsja\targeted", "nesscore\targeted", "qeba\targeted", "square\untargeted", "surfree\untargeted")

foreach ($path in $paths) {
    $command = @"
D:\Tool\SoftWare\Anaconda\python.exe -u D:\WorkSpace\progamme\Codefield\Code_Py\ccs_23_oars_stateful_attacks\main.py --config configs\cifar\blacklight\$path\adaptive\config.json --start_idx 0 --num_images 1 > output\cifar10.txt 2>&1
"@
    $command1 = @"
D:\Tool\SoftWare\Anaconda\python.exe D:\WorkSpace\progamme\Codefield\Code_Py\ccs_23_oars_stateful_attacks\main.py --config configs\cifar\blacklight\$path\adaptive\config.json --start_idx 0 --num_images 1
"@
    Write-Output "Excuting command: $command1"
    Invoke-Expression $command1
}