import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
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


def parameter(actions, name):
    for action in actions or []:
        for parameter in action.get("parameters") or []:
            if parameter.get("name") == name:
                return str(parameter.get("value") or "")
    return ""


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
