import base64
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


class CiError(RuntimeError):
    pass


def request(method, url, headers=None, body=None, content_type=None):
    request_headers = dict(headers or {})
    if content_type:
        request_headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8", errors="replace")
        raise CiError(f"HTTP {exc.code} from {url}: {data[:800]}") from exc


def json_request(method, url, headers=None, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    _, _, data = request(method, url, headers, body, "application/json" if body else None)
    return None if not data else json.loads(data.decode("utf-8"))


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


def multipart(fields, file_field, file_path):
    boundary = "----devsh-ci-" + os.urandom(12).hex()
    parts = []
    for name, value in fields.items():
        parts += [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
            str(value).encode(),
            b"\r\n",
        ]
    path = Path(file_path)
    parts += [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n".encode(),
        b"Content-Type: application/zip\r\n\r\n",
        path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


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


def wait_workflow(repo, workflow_file, branch, sha, headers, workflow_name, timeout_seconds=21600):
    deadline = time.monotonic() + timeout_seconds
    query = urllib.parse.urlencode({"branch": branch, "event": "push", "head_sha": sha, "per_page": 10})
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs?{query}"
    while time.monotonic() < deadline:
        runs = json_request("GET", url, headers).get("workflow_runs", [])
        runs = [run for run in runs if run.get("name") == workflow_name and run.get("head_sha") == sha]
        runs.sort(key=lambda run: int(run.get("run_attempt") or 0), reverse=True)
        if runs and runs[0].get("status") == "completed":
            return runs[0]
        print(f"Waiting for {workflow_name} workflow for {sha} to complete...")
        time.sleep(60)
    raise CiError(f"Timed out waiting for {workflow_name} workflow for {sha}.")


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


def start_file_job(base, headers, job, fields, file_field, file_path):
    content_type, body = multipart(fields, file_field, file_path)
    _, response_headers, _ = jpost(base, headers, f"/{job_path(job)}/buildWithParameters", body, content_type)
    if not response_headers.get("Location"):
        raise CiError(f"Jenkins did not return a queue location for {job}.")
    return wait_queue(base, headers, response_headers["Location"], job)


def wait_queue(base, headers, queue_url, job):
    for _ in range(360):
        item = jget(base, headers, f"{queue_url}/api/json")
        if item.get("cancelled"):
            raise CiError(f"Jenkins queue item for {job} was cancelled.")
        if (item.get("executable") or {}).get("number"):
            return int(item["executable"]["number"])
        time.sleep(5)
    raise CiError(f"Timed out waiting for Jenkins queue item for {job}.")


def wait_build(base, headers, job, number):
    path = job_path(job)
    for _ in range(600):
        build = jget(base, headers, f"/{path}/{number}/api/json?tree=building,result")
        if not build.get("building"):
            return str(build.get("result") or "")
        time.sleep(30)
    raise CiError(f"Timed out waiting for Jenkins build {job} #{number}.")
