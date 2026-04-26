param(
  [Parameter(Mandatory = $true)][string]$JenkinsUrl,
  [Parameter(Mandatory = $true)][string]$JenkinsUser,
  [Parameter(Mandatory = $true)][string]$JenkinsToken,
  [Parameter(Mandatory = $true)][string]$GithubToken,
  [Parameter(Mandatory = $true)][string]$PackagePath,
  [Parameter(Mandatory = $true)][string]$Repository,
  [Parameter(Mandatory = $true)][string]$Branch,
  [Parameter(Mandatory = $true)][string]$Sha,
  [Parameter(Mandatory = $true)][string]$SourceRunId,
  [Parameter(Mandatory = $true)][string]$SourceRunAttempt,
  [Parameter(Mandatory = $true)][string]$SourceWorkflow,
  [Parameter(Mandatory = $true)][ValidateSet("both", "public", "private")][string]$SceneSet,
  [Parameter(Mandatory = $true)][string]$Publish
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ($Branch -ne "ptCLI") {
  throw "Jenkins DITT trigger only accepts ptCLI."
}
if ($Repository -ne "Devsh-Graphics-Programming/Nabla") {
  throw "Jenkins DITT trigger only accepts Devsh-Graphics-Programming/Nabla."
}
if ($Sha -notmatch "^[0-9a-fA-F]{40}$") {
  throw "Invalid source SHA."
}
if (-not (Test-Path -LiteralPath $PackagePath)) {
  throw "Package does not exist: $PackagePath"
}
if ([string]::IsNullOrWhiteSpace($JenkinsUser) -or [string]::IsNullOrWhiteSpace($JenkinsToken)) {
  throw "Jenkins credentials are not configured."
}

$JenkinsUrl = $JenkinsUrl.TrimEnd("/")
$basicBytes = [System.Text.Encoding]::ASCII.GetBytes("${JenkinsUser}:${JenkinsToken}")
$jenkinsHeaders = @{
  Authorization = "Basic $([Convert]::ToBase64String($basicBytes))"
}
$githubHeaders = @{
  Authorization = "Bearer $GithubToken"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
}

function Join-JenkinsUrl {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ($Path.StartsWith("http://") -or $Path.StartsWith("https://")) {
    return $Path
  }
  if (-not $Path.StartsWith("/")) {
    $Path = "/" + $Path
  }
  return $JenkinsUrl + $Path
}

function Get-JenkinsJobPath {
  param([Parameter(Mandatory = $true)][string]$Job)
  return (($Job -split "/") | ForEach-Object { "job/$([System.Uri]::EscapeDataString($_))" }) -join "/"
}

function Invoke-JenkinsJson {
  param([Parameter(Mandatory = $true)][string]$Path)
  return Invoke-RestMethod -Method Get -Uri (Join-JenkinsUrl $Path) -Headers $jenkinsHeaders
}

function Get-JenkinsCrumbHeader {
  try {
    $crumb = Invoke-JenkinsJson -Path "/crumbIssuer/api/json"
    return @{ "$($crumb.crumbRequestField)" = [string]$crumb.crumb }
  } catch {
    return @{}
  }
}

$crumbHeader = Get-JenkinsCrumbHeader

function Invoke-JenkinsPost {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [hashtable]$Body = @{}
  )
  $headers = @{} + $jenkinsHeaders + $crumbHeader
  return Invoke-WebRequest -Method Post -Uri (Join-JenkinsUrl $Path) -Headers $headers -Body $Body -SkipHttpErrorCheck
}

function Invoke-JenkinsMultipartPost {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][hashtable]$Form
  )
  $headers = @{} + $jenkinsHeaders + $crumbHeader
  return Invoke-WebRequest -Method Post -Uri (Join-JenkinsUrl $Path) -Headers $headers -Form $Form -SkipHttpErrorCheck
}

function Get-ParameterValue {
  param(
    [object[]]$Actions,
    [Parameter(Mandatory = $true)][string]$Name
  )
  foreach ($action in @($Actions)) {
    foreach ($parameter in @($action.parameters)) {
      if ($parameter.name -eq $Name) {
        return [string]$parameter.value
      }
    }
  }
  return ""
}

function Test-SourceMatch {
  param([object[]]$Actions)
  $repo = Get-ParameterValue -Actions $Actions -Name "SOURCE_REPOSITORY"
  $branchName = Get-ParameterValue -Actions $Actions -Name "SOURCE_BRANCH"
  $runId = Get-ParameterValue -Actions $Actions -Name "SOURCE_RUN_ID"
  $runAttempt = Get-ParameterValue -Actions $Actions -Name "SOURCE_RUN_ATTEMPT"
  return ($repo -eq $Repository -and $branchName -eq $Branch -and -not ($runId -eq $SourceRunId -and $runAttempt -eq $SourceRunAttempt))
}

function Set-CommitStatus {
  param(
    [Parameter(Mandatory = $true)][string]$Context,
    [Parameter(Mandatory = $true)][ValidateSet("error", "failure", "pending", "success")][string]$State,
    [Parameter(Mandatory = $true)][string]$TargetUrl,
    [Parameter(Mandatory = $true)][string]$Description
  )
  if ($Description.Length -gt 140) {
    $Description = $Description.Substring(0, 140)
  }
  $body = @{
    state = $State
    target_url = $TargetUrl
    description = $Description
    context = $Context
  } | ConvertTo-Json -Depth 4
  Invoke-RestMethod `
    -Method Post `
    -Uri "https://api.github.com/repos/$Repository/statuses/$Sha" `
    -Headers $githubHeaders `
    -ContentType "application/json" `
    -Body $body | Out-Null
}

function Stop-OlderJenkinsRuns {
  param([Parameter(Mandatory = $true)][string]$Job)
  $jobPath = Get-JenkinsJobPath -Job $Job
  $tree = [System.Uri]::EscapeDataString("builds[number,building,result,actions[parameters[name,value]]]")
  $jobInfo = Invoke-JenkinsJson -Path "/$jobPath/api/json?tree=$tree"
  $stopped = @()
  foreach ($build in @($jobInfo.builds)) {
    if ($build.building -and (Test-SourceMatch -Actions $build.actions)) {
      Write-Host "Stopping superseded Jenkins build $Job #$($build.number)."
      $response = Invoke-JenkinsPost -Path "/$jobPath/$($build.number)/stop"
      if ($response.StatusCode -lt 200 -or $response.StatusCode -gt 399) {
        throw "Failed to stop Jenkins build $Job #$($build.number): HTTP $($response.StatusCode)."
      }
      $stopped += "#$($build.number)"
    }
  }

  $queueTree = [System.Uri]::EscapeDataString("items[id,task[fullName],actions[parameters[name,value]]]")
  $queue = Invoke-JenkinsJson -Path "/queue/api/json?tree=$queueTree"
  foreach ($item in @($queue.items)) {
    if ($item.task.fullName -eq $Job -and (Test-SourceMatch -Actions $item.actions)) {
      Write-Host "Cancelling superseded Jenkins queue item $($item.id) for $Job."
      $response = Invoke-JenkinsPost -Path "/queue/cancelItem?id=$($item.id)"
      if ($response.StatusCode -lt 200 -or $response.StatusCode -gt 399) {
        throw "Failed to cancel Jenkins queue item $($item.id): HTTP $($response.StatusCode)."
      }
      $stopped += "queue:$($item.id)"
    }
  }
  return $stopped
}

function Wait-JenkinsExecutable {
  param(
    [Parameter(Mandatory = $true)][string]$QueueUrl,
    [Parameter(Mandatory = $true)][string]$Job
  )
  $deadline = (Get-Date).AddMinutes(30)
  while ((Get-Date) -lt $deadline) {
    $queueItem = Invoke-JenkinsJson -Path "$QueueUrl/api/json"
    if ($queueItem.cancelled) {
      throw "Jenkins queue item for $Job was cancelled before it started."
    }
    if ($queueItem.executable -and $queueItem.executable.number) {
      return [int]$queueItem.executable.number
    }
    Start-Sleep -Seconds 5
  }
  throw "Timed out waiting for Jenkins queue item for $Job."
}

function Wait-JenkinsBuild {
  param(
    [Parameter(Mandatory = $true)][string]$Job,
    [Parameter(Mandatory = $true)][int]$BuildNumber
  )
  $jobPath = Get-JenkinsJobPath -Job $Job
  $deadline = (Get-Date).AddMinutes(300)
  while ((Get-Date) -lt $deadline) {
    $build = Invoke-JenkinsJson -Path "/$jobPath/$BuildNumber/api/json?tree=building,result,url,duration,description"
    if (-not $build.building) {
      return $build
    }
    Start-Sleep -Seconds 30
  }
  throw "Timed out waiting for Jenkins build $Job #$BuildNumber."
}

function Start-DittJob {
  param(
    [Parameter(Mandatory = $true)][ValidateSet("public", "private")][string]$Suite
  )
  $job = "ci/ditt/real/ex40-$Suite"
  $context = "jenkins/ditt-$Suite"
  $actionsUrl = "https://github.com/$Repository/actions/runs/$SourceRunId"
  if ($SourceRunAttempt -and $SourceRunAttempt -ne "1") {
    $actionsUrl = "$actionsUrl/attempts/$SourceRunAttempt"
  }

  Set-CommitStatus -Context $context -State pending -TargetUrl $actionsUrl -Description "Waiting for Jenkins $Suite DITT run."
  Stop-OlderJenkinsRuns -Job $job | Out-Null

  $jobPath = Get-JenkinsJobPath -Job $job
  $form = @{
    EX40_PACKAGE_FILE = Get-Item -LiteralPath $PackagePath
    FAIL_ON_RENDER_FAILURE = "false"
    PUBLISH = $Publish.ToLowerInvariant()
    SOURCE_REPOSITORY = $Repository
    SOURCE_BRANCH = $Branch
    SOURCE_SHA = $Sha
    SOURCE_RUN_ID = $SourceRunId
    SOURCE_RUN_ATTEMPT = $SourceRunAttempt
    SOURCE_WORKFLOW = $SourceWorkflow
  }
  $response = Invoke-JenkinsMultipartPost -Path "/$jobPath/buildWithParameters" -Form $form
  if ($response.StatusCode -lt 200 -or $response.StatusCode -gt 399) {
    Set-CommitStatus -Context $context -State failure -TargetUrl $actionsUrl -Description "Jenkins refused to start the $Suite DITT run."
    throw "Jenkins refused to start ${job}: HTTP $($response.StatusCode)."
  }

  $queueUrl = [string](@($response.Headers.Location) | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($queueUrl)) {
    Set-CommitStatus -Context $context -State failure -TargetUrl $actionsUrl -Description "Jenkins did not return a queue URL."
    throw "Jenkins did not return a queue URL for $job."
  }

  $buildNumber = Wait-JenkinsExecutable -QueueUrl $queueUrl -Job $job
  $buildUrl = "$JenkinsUrl/$jobPath/$buildNumber/"
  Set-CommitStatus -Context $context -State pending -TargetUrl $buildUrl -Description "Jenkins $Suite DITT build #$buildNumber is running."
  Write-Host "Started Jenkins $job #${buildNumber}: $buildUrl"

  $build = Wait-JenkinsBuild -Job $job -BuildNumber $buildNumber
  $result = [string]$build.result
  if ($result -eq "SUCCESS") {
    Set-CommitStatus -Context $context -State success -TargetUrl $buildUrl -Description "Jenkins $Suite DITT succeeded."
    return @{ job = $job; number = $buildNumber; result = $result; url = $buildUrl }
  }
  if ($result -eq "UNSTABLE") {
    Set-CommitStatus -Context $context -State success -TargetUrl $buildUrl -Description "Jenkins $Suite DITT is unstable because the report has failures."
    return @{ job = $job; number = $buildNumber; result = $result; url = $buildUrl }
  }

  Set-CommitStatus -Context $context -State failure -TargetUrl $buildUrl -Description "Jenkins $Suite DITT finished with $result."
  throw "Jenkins $job #$buildNumber finished with $result."
}

$suites = switch ($SceneSet) {
  "both" { @("public", "private") }
  "public" { @("public") }
  "private" { @("private") }
}

$results = foreach ($suite in $suites) {
  Start-DittJob -Suite $suite
}

Write-Host "Jenkins DITT results:"
foreach ($result in $results) {
  Write-Host "$($result.job) #$($result.number) $($result.result) $($result.url)"
}
