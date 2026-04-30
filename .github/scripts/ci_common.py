import base64
import fnmatch
import json
import os
import re
import shutil
import socket
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


class CiError(RuntimeError):
    pass


TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def request(method, url, headers=None, body=None, content_type=None):
    request_headers = dict(headers or {})
    if content_type:
        request_headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    attempts = 6 if method.upper() == "GET" else 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            if attempt < attempts and exc.code in TRANSIENT_HTTP_CODES:
                print(f"Transient HTTP {exc.code}; retrying request {attempt}/{attempts}.")
                time.sleep(min(5 * attempt, 30))
                continue
            data = exc.read().decode("utf-8", errors="replace")
            raise CiError(f"HTTP {exc.code} from {url}: {data[:800]}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            if attempt < attempts:
                print(f"Transient request failure; retrying request {attempt}/{attempts}.")
                time.sleep(min(5 * attempt, 30))
                continue
            raise CiError(f"HTTP request failed for {url}: {exc}") from exc


def json_request(method, url, headers=None, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    _, _, data = request(method, url, headers, body, "application/json" if body else None)
    return None if not data else json.loads(data.decode("utf-8"))


def text_request(method, url, headers=None):
    _, _, data = request(method, url, headers)
    return data.decode("utf-8", errors="replace")


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def basic_headers(user, token):
    raw = base64.b64encode(f"{user}:{token}".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {raw}"}


def join_url(base, path):
    if path.startswith(("http://", "https://")):
        return path
    return base.rstrip("/") + (path if path.startswith("/") else "/" + path)


def job_path(job):
    return "/".join(f"job/{urllib.parse.quote(part, safe='')}" for part in job.split("/"))


def output(values):
    target = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}" for key, value in values.items()]
    if not target:
        print("\n".join(lines))
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def choice(value, allowed, name):
    if value not in allowed:
        raise CiError(f"Invalid {name}: {value}.")
    return value


def require_sha(value):
    value = str(value or "")
    if len(value) != 40 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise CiError("Invalid source SHA.")
    return value.lower()


def bool_text(value):
    value = str(value).lower()
    if value not in {"true", "false"}:
        raise CiError(f"Invalid boolean value: {value}.")
    return value


def add_args(parser, names, required=True, default=None):
    for name in names:
        parser.add_argument(f"--{name}", required=required, default=default)


def multipart_files(fields, files):
    boundary = "----devsh-ci-" + os.urandom(12).hex()
    parts = []
    for name, value in fields.items():
        parts += [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
            str(value).encode(),
            b"\r\n",
        ]
    for file_field, file_path in files:
        path = Path(file_path)
        parts += [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n".encode(),
            b"Content-Type: application/zip\r\n\r\n",
            path.read_bytes(),
            b"\r\n",
        ]
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def multipart(fields, file_field, file_path):
    return multipart_files(fields, [(file_field, file_path)])


def extract_single_tar(artifact_dir, pattern):
    artifact_dir = Path(artifact_dir)
    matches = list(artifact_dir.glob(pattern))
    if len(matches) != 1:
        raise CiError(f"Expected one tar matching {pattern}, found {len(matches)}.")
    base = artifact_dir.resolve()
    with tarfile.open(matches[0]) as tar:
        members = tar.getmembers()
        for member in members:
            target = (artifact_dir / member.name).resolve()
            if os.path.commonpath([str(base), str(target)]) != str(base):
                raise CiError("Tar archive contains a path outside the artifact directory.")
            if member.issym() or member.islnk():
                raise CiError("Tar archive contains links, refusing to extract.")
        try:
            tar.extractall(artifact_dir, members=members, filter="data")
        except TypeError:
            tar.extractall(artifact_dir, members=members)


def copy_contents(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise CiError(f"Missing directory: {src}.")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def zip_directory(src, zip_path, max_bytes):
    src = Path(src)
    zip_path = Path(zip_path)
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(src).as_posix())
    if zip_path.stat().st_size > max_bytes:
        raise CiError("Zip package is unexpectedly large.")


def jget(base, headers, path):
    return json_request("GET", join_url(base, path), headers)


def jpost(base, headers, path, body=b"", content_type="application/x-www-form-urlencoded"):
    return request("POST", join_url(base, path), headers, body, content_type)


def commit_status(repo, sha, token, context, state, target, description):
    json_request("POST", f"https://api.github.com/repos/{repo}/statuses/{sha}", github_headers(token), {
        "state": state,
        "target_url": target,
        "description": description[:140],
        "context": context,
    })


def workflow_runs_url(repo, workflow_file, branch, sha):
    query = urllib.parse.urlencode({"branch": branch, "event": "push", "head_sha": sha, "per_page": 10})
    return f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs?{query}"


def latest_workflow_run(repo, workflow_file, branch, sha, headers, workflow_name):
    runs = json_request("GET", workflow_runs_url(repo, workflow_file, branch, sha), headers).get("workflow_runs", [])
    runs = [run for run in runs if run.get("name") == workflow_name and run.get("head_sha") == sha]
    runs.sort(key=lambda run: int(run.get("run_attempt") or 0), reverse=True)
    return runs[0] if runs else None


def wait_workflow(repo, workflow_file, branch, sha, headers, workflow_name, timeout_seconds=21600):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run = latest_workflow_run(repo, workflow_file, branch, sha, headers, workflow_name)
        if run and run.get("status") == "completed":
            return run
        print(f"Waiting for {workflow_name} workflow for {sha} to complete...")
        time.sleep(60)
    raise CiError(f"Timed out waiting for {workflow_name} workflow for {sha}.")


def wait_workflow_job_artifact(repo, workflow_file, branch, sha, headers, workflow_name, job_name, artifact_pattern, timeout_seconds=21600):
    deadline = time.monotonic() + timeout_seconds
    last_log = 0
    while time.monotonic() < deadline:
        run = latest_workflow_run(repo, workflow_file, branch, sha, headers, workflow_name)
        if run:
            jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"
            jobs = json_request("GET", jobs_url, headers).get("jobs", [])
            job = next((item for item in jobs if item.get("name") == job_name), None)
            if job and job.get("conclusion") and job.get("conclusion") != "success":
                raise CiError(f"{workflow_name} job {job_name} finished with {job.get('conclusion')}.")
            artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100"
            artifacts = json_request("GET", artifacts_url, headers).get("artifacts", [])
            matches = [item for item in artifacts if fnmatch.fnmatchcase(item.get("name", ""), artifact_pattern) and not item.get("expired")]
            if job and job.get("conclusion") == "success" and matches:
                return run
            if run.get("status") == "completed" and run.get("conclusion") not in {None, "success"}:
                raise CiError(f"{workflow_name} workflow finished with {run.get('conclusion')}.")
        now = time.monotonic()
        if now - last_log >= 60:
            print(f"Waiting for {workflow_name} {job_name} and {artifact_pattern} for {sha}...")
            last_log = now
        time.sleep(30)
    raise CiError(f"Timed out waiting for {workflow_name} {job_name} artifact for {sha}.")


def cancel_older_workflow_runs(repo, token, workflow_name, branch, current_run_id):
    headers = github_headers(token)
    current = json_request("GET", f"https://api.github.com/repos/{repo}/actions/runs/{current_run_id}", headers)
    current_created = current.get("created_at", "")
    current_sha = current.get("head_sha", "")
    statuses = ["queued", "in_progress", "waiting", "requested", "pending"]
    cancelled = []
    for status in statuses:
        query = urllib.parse.urlencode({"branch": branch, "status": status, "per_page": 100})
        runs = json_request("GET", f"https://api.github.com/repos/{repo}/actions/runs?{query}", headers).get("workflow_runs", [])
        for run in runs:
            if int(run.get("id") or 0) == int(current_run_id):
                continue
            if run.get("name") != workflow_name or run.get("head_branch") != branch:
                continue
            if current_sha and run.get("head_sha") == current_sha:
                continue
            if current_created and run.get("created_at", "") >= current_created:
                continue
            json_request("POST", f"https://api.github.com/repos/{repo}/actions/runs/{run['id']}/cancel", headers)
            cancelled.append(run["id"])
            print(f"Cancelled superseded {workflow_name} run {run['id']} on {branch}.")
    if not cancelled:
        print(f"No superseded {workflow_name} runs found on {branch}.")


def parameter(actions, name):
    for action in actions or []:
        for parameter in action.get("parameters") or []:
            if parameter.get("name") == name:
                return str(parameter.get("value") or "")
    return ""


def parameters(actions):
    result = {}
    for action in actions or []:
        for param in action.get("parameters") or []:
            name = param.get("name")
            if name:
                result[name] = str(param.get("value") or "")
    return result


def superseded(actions, match, current):
    values = parameters(actions)
    return all(values.get(key) == value for key, value in match.items()) and not all(
        values.get(key) == value for key, value in current.items()
    )


def add_jenkins_crumb(base, headers):
    try:
        crumb = jget(base, headers, "/crumbIssuer/api/json")
    except CiError:
        return
    if crumb.get("crumbRequestField") and crumb.get("crumb"):
        headers[crumb["crumbRequestField"]] = crumb["crumb"]


def cancel_superseded(base, headers, job, match, current):
    path = job_path(job)
    tree = urllib.parse.quote("builds[number,building,actions[parameters[name,value]]]", safe="")
    for build in jget(base, headers, f"/{path}/api/json?tree={tree}").get("builds", []):
        if build.get("building") and superseded(build.get("actions"), match, current):
            jpost(base, headers, f"/{path}/{build['number']}/stop")
    tree = urllib.parse.quote("items[id,task[fullName],actions[parameters[name,value]]]", safe="")
    for item in jget(base, headers, f"/queue/api/json?tree={tree}").get("items", []):
        if (item.get("task") or {}).get("fullName") == job and superseded(item.get("actions"), match, current):
            jpost(base, headers, f"/queue/cancelItem?id={item['id']}")


def cancel_matching(base, headers, job, match):
    path = job_path(job)
    tree = urllib.parse.quote("builds[number,building,actions[parameters[name,value]]]", safe="")
    for build in jget(base, headers, f"/{path}/api/json?tree={tree}").get("builds", []):
        values = parameters(build.get("actions"))
        if build.get("building") and all(values.get(key) == value for key, value in match.items()):
            print(f"Stopping superseded Jenkins build {job} #{build['number']}.")
            jpost(base, headers, f"/{path}/{build['number']}/stop")
    tree = urllib.parse.quote("items[id,task[fullName],actions[parameters[name,value]]]", safe="")
    for item in jget(base, headers, f"/queue/api/json?tree={tree}").get("items", []):
        values = parameters(item.get("actions"))
        if (item.get("task") or {}).get("fullName") == job and all(values.get(key) == value for key, value in match.items()):
            print(f"Cancelling superseded Jenkins queue item {item['id']} for {job}.")
            jpost(base, headers, f"/queue/cancelItem?id={item['id']}")


def start_file_job(base, headers, job, fields, file_field, file_path):
    content_type, body = multipart(fields, file_field, file_path)
    _, response_headers, _ = jpost(base, headers, f"/{job_path(job)}/buildWithParameters", body, content_type)
    if not response_headers.get("Location"):
        raise CiError(f"Jenkins did not return a queue location for {job}.")
    return wait_queue(base, headers, response_headers["Location"], job)


def start_files_job(base, headers, job, fields, files):
    content_type, body = multipart_files(fields, files)
    _, response_headers, _ = jpost(base, headers, f"/{job_path(job)}/buildWithParameters", body, content_type)
    if not response_headers.get("Location"):
        raise CiError(f"Jenkins did not return a queue location for {job}.")
    return wait_queue(base, headers, response_headers["Location"], job)


def stop_build(base, headers, job, number):
    try:
        jpost(base, headers, f"/{job_path(job)}/{number}/stop")
    except CiError as exc:
        print(f"Could not stop Jenkins build {job} #{number}: {exc}")


def redact_console(text):
    patterns = [
        r"(?i)(authorization:\s*)(basic|bearer)\s+\S+",
        r"(?i)((?:token|password|secret|cookie|access[_-]?key)\s*[=:]\s*)\S+",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(aws_secret_access_key\s*[=:]\s*)\S+",
    ]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, lambda match: match.group(1) + "***" if match.lastindex else "***", redacted)
    return redacted


def print_console_tail(base, headers, job, number, lines=200):
    try:
        text = text_request("GET", join_url(base, f"/{job_path(job)}/{number}/consoleText"), headers)
    except CiError as exc:
        print(f"Could not fetch Jenkins console tail for {job} #{number}: {exc}")
        return
    tail = "\n".join(redact_console(text).splitlines()[-lines:])
    print(f"--- Jenkins console tail: {job} #{number} ---")
    print(tail)
    print("--- End Jenkins console tail ---")


def wait_queue(base, headers, queue_url, job):
    for _ in range(360):
        item = jget(base, headers, f"{queue_url}/api/json")
        if item.get("cancelled"):
            raise CiError(f"Jenkins queue item for {job} was cancelled.")
        if (item.get("executable") or {}).get("number"):
            return int(item["executable"]["number"])
        time.sleep(5)
    raise CiError(f"Timed out waiting for Jenkins queue item for {job}.")


def wait_build(base, headers, job, number, timeout_seconds=14400):
    path = job_path(job)
    deadline = time.monotonic() + timeout_seconds
    last_log = 0
    while time.monotonic() < deadline:
        build = jget(base, headers, f"/{path}/{number}/api/json?tree=building,result")
        if not build.get("building"):
            return str(build.get("result") or "")
        now = time.monotonic()
        if now - last_log >= 300:
            print(f"Jenkins {job} #{number} is still running.")
            last_log = now
        time.sleep(30)
    raise CiError(f"Timed out waiting for Jenkins build {job} #{number}.")
