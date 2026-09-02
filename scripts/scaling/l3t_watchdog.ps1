# Windows-side watchdog for the local 3-topology run.
#
# WHY IT LIVES OUTSIDE WSL. The first attempt died 12 minutes in when the WSL
# service dropped (Wsl/Service/0x8007274c, hit four times in an hour). A
# supervisor inside WSL goes down with the VM, so the retry loop in
# local_3t_fixed.sh handles process crashes and this handles VM death.
#
# It relaunches only when ALL of these hold:
#   * the run is not finished (no l3t_fix_head.done)
#   * no experiment.py is alive inside WSL
#   * the heartbeat is stale
# so it cannot stampede a run that is merely slow to compile -- the reset
# compile alone took 240 s.
#
#   powershell -ExecutionPolicy Bypass -File scripts\scaling\l3t_watchdog.ps1
#   powershell ... -File ...\l3t_watchdog.ps1 -Once      (single check, for a scheduled task)
param(
    [int]$StaleMinutes = 25,
    [int]$IntervalSeconds = 300,
    [switch]$Once
)

$Repo = "C:\Users\smirn\Desktop\robot_learning_ip"
$Out = Join-Path $Repo "experiments\local_3t"
$script:wslMisses = 0
$WatchLog = Join-Path $Out "l3t_watchdog.log"
$RunLog = Join-Path $Out "l3t_fixed_run.log"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    $line | Tee-Object -FilePath $WatchLog -Append
}

function TrainingAlive {
    # -f matches the command line, not the process name (which is "python").
    $n = (wsl -d Ubuntu -- bash -lc "pgrep -cf 'experiment.py' 2>/dev/null || echo 0") 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $n) { return $null }   # WSL itself down
    return ([int]($n -replace '\D', '') -gt 0)
}

function Finished {
    # ARM 2 was renamed l3t_fix_head -> l3t_fix_head_rx (rx_p3f0 recipe per the
    # 2026-08-29 plan). It is the last arm, so its .done means the run is over.
    Test-Path (Join-Path $Out "l3t_fix_head_rx.done")
}

function HeartbeatAgeMinutes {
    $beats = Get-ChildItem (Join-Path $Out "*.heartbeat") -ErrorAction SilentlyContinue
    if (-not $beats) { return 9999 }
    $newest = ($beats | Sort-Object LastWriteTime -Descending)[0]
    return [int]((Get-Date) - $newest.LastWriteTime).TotalMinutes
}

function Relaunch {
    Note "RELAUNCHING (each arm resumes from its newest checkpoint)"
    $cmd = "cd /mnt/c/Users/smirn/Desktop/robot_learning_ip && setsid nohup bash scripts/scaling/local_3t_fixed.sh >> experiments/local_3t/l3t_fixed_run.log 2>&1 < /dev/null &"
    wsl -d Ubuntu -- bash -lc $cmd 2>$null | Out-Null
    Start-Sleep -Seconds 20
}

do {
    if (Finished) { Note "run COMPLETE (l3t_fix_head.done) -- watchdog exiting"; break }

    $alive = TrainingAlive
    $age = HeartbeatAgeMinutes
    if ($null -eq $alive) {
        # WSL WEDGED, not merely busy. On 2026-08-29 the VM stopped responding
        # for 40 minutes while the GPU sat at 2% and both jobs inside were hung;
        # this watchdog detected it correctly every 10 min and could do nothing,
        # because it only knew how to restart a dead PROCESS. Two consecutive
        # misses now restart the VM -- everything inside is already hung, and the
        # training arm resumes from its newest checkpoint.
        $script:wslMisses = $script:wslMisses + 1
        Note ("WSL not responding (miss {0})" -f $script:wslMisses)
        if ($script:wslMisses -ge 2) {
            Note "WSL wedged -- shutting the VM down so it restarts clean"
            wsl --shutdown 2>$null | Out-Null
            Start-Sleep -Seconds 20
            $script:wslMisses = 0
            if (-not (Finished)) { Relaunch }
        }
    }
    elseif ($alive) {
        $script:wslMisses = 0
        Note ("training alive, heartbeat {0} min old" -f $age)
    }
    elseif ($age -lt $StaleMinutes) {
        # No process yet but the heartbeat is fresh: an arm is between attempts,
        # or compiling before it writes anything. Do not stampede it.
        Note ("no process, but heartbeat only {0} min old -- waiting" -f $age)
    }
    else {
        Note ("DEAD: no experiment.py and heartbeat {0} min old (>{1})" -f $age, $StaleMinutes)
        Relaunch
    }
    if (-not $Once) { Start-Sleep -Seconds $IntervalSeconds }
} while (-not $Once)
