#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


VALID_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_PACKAGE_BYTES = 200 * 1024 * 1024
SOURCE_WORKFLOW = "Build"
BUILD_WORKFLOW_FILE = "build-nabla.yml"
EXPECTED_REPOSITORY = "Devsh-Graphics-Programming/Nabla"
EXPECTED_BRANCH = "ptCLI"


class FatalError(RuntimeError):
    pass


def env(name, required=True, default=""):
    value = os.environ.get(name, default)
    if required and not value:
        raise FatalError(f"Missing environment variable: {name}")
    return value


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def http_request(method, url, headers=None, body=None, content_type=None, timeout=60):
    request_headers = dict(headers or {})
    if content_type:
        request_headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8", errors="replace")
        raise FatalError(f"HTTP {exc.code} from {url}: {data[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise FatalError(f"Request failed for {url}: {exc}") from exc


def get_json(url, headers):
    _, _, data = http_request("GET", url, headers=headers)
    return json.loads(data.decode("utf-8"))


def post_json(url, headers, payload):
    body = json.dumps(payload).encode("utf-8")
    _, _, data = http_request("POST", url, headers=headers, body=body, content_type="application/json")
    if not data:
        return None
    return json.loads(data.decode("utf-8"))


def write_github_output(values):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        for key, value in values.items():
            print(f"{key}={value}")
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def find_source_run(repository, branch, sha, token):
    deadline = time.monotonic() + 6 * 60 * 60
    headers = github_headers(token)
    query = urllib.parse.urlencode(
        {
            "branch": branch,
            "event": "push",
            "head_sha": sha,
            "per_page": "10",
        }
    )
    url = f"https://api.github.com/repos/{repository}/actions/workflows/{BUILD_WORKFLOW_FILE}/runs?{query}"
    while time.monotonic() < deadline:
        response = get_json(url, headers)
        runs = [
            run
            for run in response.get("workflow_runs", [])
            if str(run.get("head_sha")) == sha and str(run.get("name")) == SOURCE_WORKFLOW
        ]
        runs.sort(key=lambda run: int(run.get("run_attempt") or 0), reverse=True)
        if runs:
            run = runs[0]
            if run.get("status") == "completed":
                return run
        print(f"Waiting for {SOURCE_WORKFLOW} workflow for {sha} to complete...")
        time.sleep(60)
    raise FatalError(f"Timed out waiting for {SOURCE_WORKFLOW} workflow for {sha}.")


def validate_source_run(run, config, scene_set, publish):
    branch = str(run.get("head_branch") or "")
    sha = str(run.get("head_sha") or "")
    conclusion = str(run.get("conclusion") or "")
    workflow = str(run.get("name") or "")
    run_id = str(run.get("id") or "")
    attempt = str(run.get("run_attempt") or "")

    if branch != EXPECTED_BRANCH:
        raise FatalError(f"Path Tracer Jenkins workflow only accepts Build runs from {EXPECTED_BRANCH}. Got '{branch}'.")
    if conclusion != "success":
        raise FatalError(f"Build workflow run {run_id} is not green. Conclusion: {conclusion}.")
    if not VALID_SHA_RE.match(sha):
        raise FatalError(f"Build workflow run {run_id} does not expose a valid head SHA.")
    if workflow != SOURCE_WORKFLOW:
        raise FatalError(f"Build workflow run {run_id} has unexpected workflow name: {workflow}.")

    return {
        "run_id": run_id,
        "build_config": config,
        "scene_set": scene_set,
        "publish": publish,
        "branch": branch,
        "sha": sha,
        "run_attempt": attempt,
        "workflow": workflow,
    }


def resolve_source(args):
    token = env("GH_TOKEN")
    repository = args.repository
    event_name = args.event_name
    if repository != EXPECTED_REPOSITORY:
        raise FatalError(f"Path Tracer Jenkins workflow only accepts {EXPECTED_REPOSITORY}.")

    headers = github_headers(token)
    if event_name == "push":
        run = find_source_run(repository, args.branch, args.sha, token)
        values = validate_source_run(run, "RelWithDebInfo", "both", "true")
    else:
        run = get_json(f"https://api.github.com/repos/{repository}/actions/runs/{args.run_id}", headers)
        values = validate_source_run(run, args.build_config, args.scene_set, str(args.publish).lower())

    write_github_output(values)
    print(f"Resolved Build run {values['run_id']} on {values['branch']} at {values['sha']}.")


def first_file(root, pattern):
    matches = sorted(Path(root).glob(pattern))
    return matches[0] if matches else None


def copy_tree_contents(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def prepare_package(args):
    artifact_dir = Path(args.artifact_dir)
    tar_path = first_file(artifact_dir, "*-install.tar")
    if tar_path is None:
        raise FatalError("Install artifact tar was not downloaded.")
    with tarfile.open(tar_path, "r") as tar:
        try:
            tar.extractall(artifact_dir, filter="data")
        except TypeError:
            tar.extractall(artifact_dir)

    install_root = artifact_dir / "build-ct" / "install"
    if not install_root.exists():
        raise FatalError(f"Install root was not found: {install_root}")

    config = args.build_config
    config_root = install_root if config == "Release" else install_root / config.lower()
    runtime_root = config_root / "runtime"
    ex40_root = config_root / "exe" / "examples_tests" / "40_PathTracer" / "bin"
    if not runtime_root.exists():
        raise FatalError(f"Runtime directory was not found: {runtime_root}")
    if not ex40_root.exists():
        raise FatalError(f"EX40 runtime directory was not found: {ex40_root}")

    package_root = Path(args.package_root)
    zip_path = Path(args.zip_path)
    shutil.rmtree(package_root, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()

    prefix = Path() if config == "Release" else Path(config.lower())
    runtime_dest = package_root / prefix / "runtime"
    ex40_dest = package_root / prefix / "exe" / "examples_tests" / "40_PathTracer" / "bin"
    copy_tree_contents(runtime_root, runtime_dest)
    copy_tree_contents(ex40_root, ex40_dest)

    if not list(package_root.rglob("40_pathtracer*.exe")):
        raise FatalError("40_pathtracer executable was not found in the package.")
    if not list(package_root.rglob("Nabla_*.dll")):
        raise FatalError("Nabla runtime DLL was not found in the package.")
    if not list(package_root.rglob("dxcompiler.dll")):
        raise FatalError("dxcompiler.dll was not found in the package.")
    exe = sorted(package_root.rglob("40_pathtracer*.exe"))[0]
    if not (exe.parent / "report").exists():
        raise FatalError("EX40 report template was not found in the package.")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root).as_posix())
    size = zip_path.stat().st_size
    if size > MAX_PACKAGE_BYTES:
        raise FatalError(f"EX40 package is unexpectedly large: {size} bytes.")

    write_github_output({"zip_path": str(zip_path)})
    print(f"Prepared EX40 runtime package: {size} bytes.")


def jenkins_headers(user, token):
    auth = base64.b64encode(f"{user}:{token}".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {auth}"}


def join_jenkins_url(base, path):
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return base.rstrip("/") + path


def jenkins_job_path(job):
    return "/".join(f"job/{urllib.parse.quote(part, safe='')}" for part in job.split("/"))


def jenkins_json(base, headers, path):
    return get_json(join_jenkins_url(base, path), headers)


def jenkins_crumb(base, headers):
    try:
        crumb = jenkins_json(base, headers, "/crumbIssuer/api/json")
        return {str(crumb["crumbRequestField"]): str(crumb["crumb"])}
    except Exception:
        return {}


def jenkins_post(base, headers, path, body=b"", content_type="application/x-www-form-urlencoded"):
    return http_request("POST", join_jenkins_url(base, path), headers=headers, body=body, content_type=content_type)


def multipart_body(fields, files):
    boundary = f"----devsh-path-tracer-{int(time.time() * 1000)}"
    chunks = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files.items():
        path = Path(path)
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode("utf-8")
        )
        chunks.append(b"Content-Type: application/zip\r\n\r\n")
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


def parameter_value(actions, name):
    for action in actions or []:
        for parameter in action.get("parameters") or []:
            if parameter.get("name") == name:
                return str(parameter.get("value") or "")
    return ""


def source_matches(actions, repository, branch, run_id, run_attempt):
    repo = parameter_value(actions, "SOURCE_REPOSITORY")
    source_branch = parameter_value(actions, "SOURCE_BRANCH")
    source_run_id = parameter_value(actions, "SOURCE_RUN_ID")
    source_run_attempt = parameter_value(actions, "SOURCE_RUN_ATTEMPT")
    return repo == repository and source_branch == branch and not (
        source_run_id == run_id and source_run_attempt == run_attempt
    )


def set_commit_status(repository, sha, token, context, state, target_url, description):
    if len(description) > 140:
        description = description[:140]
    post_json(
        f"https://api.github.com/repos/{repository}/statuses/{sha}",
        github_headers(token),
        {
            "state": state,
            "target_url": target_url,
            "description": description,
            "context": context,
        },
    )


def stop_older_jenkins_runs(base, headers, job, repository, branch, run_id, run_attempt):
    job_path = jenkins_job_path(job)
    tree = urllib.parse.quote("builds[number,building,result,actions[parameters[name,value]]]", safe="")
    job_info = jenkins_json(base, headers, f"/{job_path}/api/json?tree={tree}")
    for build in job_info.get("builds") or []:
        if build.get("building") and source_matches(build.get("actions"), repository, branch, run_id, run_attempt):
            print(f"Stopping superseded Jenkins build {job} #{build.get('number')}.")
            status, _, _ = jenkins_post(base, headers, f"/{job_path}/{build.get('number')}/stop")
            if status < 200 or status > 399:
                raise FatalError(f"Failed to stop Jenkins build {job} #{build.get('number')}: HTTP {status}.")

    queue_tree = urllib.parse.quote("items[id,task[fullName],actions[parameters[name,value]]]", safe="")
    queue = jenkins_json(base, headers, f"/queue/api/json?tree={queue_tree}")
    for item in queue.get("items") or []:
        task = item.get("task") or {}
        if task.get("fullName") == job and source_matches(item.get("actions"), repository, branch, run_id, run_attempt):
            print(f"Cancelling superseded Jenkins queue item {item.get('id')} for {job}.")
            status, _, _ = jenkins_post(base, headers, f"/queue/cancelItem?id={item.get('id')}")
            if status < 200 or status > 399:
                raise FatalError(f"Failed to cancel Jenkins queue item {item.get('id')}: HTTP {status}.")


def wait_jenkins_executable(base, headers, queue_url, job):
    deadline = time.monotonic() + 30 * 60
    while time.monotonic() < deadline:
        item = jenkins_json(base, headers, f"{queue_url}/api/json")
        if item.get("cancelled"):
            raise FatalError(f"Jenkins queue item for {job} was cancelled before it started.")
        executable = item.get("executable") or {}
        if executable.get("number"):
            return int(executable["number"])
        time.sleep(5)
    raise FatalError(f"Timed out waiting for Jenkins queue item for {job}.")


def wait_jenkins_build(base, headers, job, build_number):
    job_path = jenkins_job_path(job)
    deadline = time.monotonic() + 300 * 60
    while time.monotonic() < deadline:
        build = jenkins_json(base, headers, f"/{job_path}/{build_number}/api/json?tree=building,result,url,duration,description")
        if not build.get("building"):
            return build
        time.sleep(30)
    raise FatalError(f"Timed out waiting for Jenkins build {job} #{build_number}.")


def start_path_tracer_job(args, headers, suite):
    job = f"ci/ditt/real/ex40-{suite}"
    context = f"jenkins/path-tracer-{suite}"
    actions_url = f"https://github.com/{args.repository}/actions/runs/{args.source_run_id}"
    if args.source_run_attempt and args.source_run_attempt != "1":
        actions_url = f"{actions_url}/attempts/{args.source_run_attempt}"

    set_commit_status(args.repository, args.sha, args.github_token, context, "pending", actions_url, f"Waiting for Jenkins {suite} path tracer run.")
    stop_older_jenkins_runs(args.jenkins_url, headers, job, args.repository, args.branch, args.source_run_id, args.source_run_attempt)

    fields = {
        "FAIL_ON_RENDER_FAILURE": "false",
        "PUBLISH": args.publish.lower(),
        "SOURCE_REPOSITORY": args.repository,
        "SOURCE_BRANCH": args.branch,
        "SOURCE_SHA": args.sha,
        "SOURCE_RUN_ID": args.source_run_id,
        "SOURCE_RUN_ATTEMPT": args.source_run_attempt,
        "SOURCE_WORKFLOW": args.source_workflow,
    }
    boundary, body = multipart_body(fields, {"EX40_PACKAGE_FILE": args.package_path})
    job_path = jenkins_job_path(job)
    status, response_headers, _ = jenkins_post(
        args.jenkins_url,
        headers,
        f"/{job_path}/buildWithParameters",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    if status < 200 or status > 399:
        set_commit_status(args.repository, args.sha, args.github_token, context, "failure", actions_url, f"Jenkins refused to start the {suite} path tracer run.")
        raise FatalError(f"Jenkins refused to start {job}: HTTP {status}.")
    queue_url = response_headers.get("Location", "")
    if not queue_url:
        set_commit_status(args.repository, args.sha, args.github_token, context, "failure", actions_url, "Jenkins did not return a queue URL.")
        raise FatalError(f"Jenkins did not return a queue URL for {job}.")

    build_number = wait_jenkins_executable(args.jenkins_url, headers, queue_url, job)
    build_url = f"{args.jenkins_url.rstrip('/')}/{job_path}/{build_number}/"
    set_commit_status(args.repository, args.sha, args.github_token, context, "pending", build_url, f"Jenkins {suite} path tracer build #{build_number} is running.")
    print(f"Started Jenkins {job} #{build_number}: {build_url}")

    build = wait_jenkins_build(args.jenkins_url, headers, job, build_number)
    result = str(build.get("result") or "")
    if result == "SUCCESS":
        set_commit_status(args.repository, args.sha, args.github_token, context, "success", build_url, f"Jenkins {suite} path tracer succeeded.")
        return {"job": job, "number": build_number, "result": result, "url": build_url}
    if result == "UNSTABLE":
        set_commit_status(args.repository, args.sha, args.github_token, context, "success", build_url, f"Jenkins {suite} path tracer is unstable because the report has failures.")
        return {"job": job, "number": build_number, "result": result, "url": build_url}

    set_commit_status(args.repository, args.sha, args.github_token, context, "failure", build_url, f"Jenkins {suite} path tracer finished with {result}.")
    raise FatalError(f"Jenkins {job} #{build_number} finished with {result}.")


def trigger_jenkins(args):
    if args.branch != EXPECTED_BRANCH:
        raise FatalError(f"Path Tracer Jenkins trigger only accepts {EXPECTED_BRANCH}.")
    if args.repository != EXPECTED_REPOSITORY:
        raise FatalError(f"Path Tracer Jenkins trigger only accepts {EXPECTED_REPOSITORY}.")
    if not VALID_SHA_RE.match(args.sha):
        raise FatalError("Invalid source SHA.")
    if not Path(args.package_path).exists():
        raise FatalError(f"Package does not exist: {args.package_path}")
    if not args.jenkins_user or not args.jenkins_token:
        raise FatalError("Jenkins credentials are not configured.")

    headers = jenkins_headers(args.jenkins_user, args.jenkins_token)
    headers.update(jenkins_crumb(args.jenkins_url, headers))
    suites = ["public", "private"] if args.scene_set == "both" else [args.scene_set]
    results = [start_path_tracer_job(args, headers, suite) for suite in suites]

    print("Jenkins Path Tracer results:")
    for result in results:
        print(f"{result['job']} #{result['number']} {result['result']} {result['url']}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve-source")
    resolve.add_argument("--repository", required=True)
    resolve.add_argument("--event-name", required=True)
    resolve.add_argument("--branch", default="")
    resolve.add_argument("--sha", default="")
    resolve.add_argument("--run-id", default="")
    resolve.add_argument("--build-config", default="RelWithDebInfo", choices=["RelWithDebInfo", "Release", "Debug"])
    resolve.add_argument("--scene-set", default="both", choices=["both", "public", "private"])
    resolve.add_argument("--publish", default="true")
    resolve.set_defaults(func=resolve_source)

    package = subparsers.add_parser("prepare-package")
    package.add_argument("--artifact-dir", required=True)
    package.add_argument("--build-config", required=True, choices=["RelWithDebInfo", "Release", "Debug"])
    package.add_argument("--package-root", required=True)
    package.add_argument("--zip-path", required=True)
    package.set_defaults(func=prepare_package)

    trigger = subparsers.add_parser("trigger-jenkins")
    trigger.add_argument("--jenkins-url", required=True)
    trigger.add_argument("--jenkins-user", required=True)
    trigger.add_argument("--jenkins-token", required=True)
    trigger.add_argument("--github-token", required=True)
    trigger.add_argument("--package-path", required=True)
    trigger.add_argument("--repository", required=True)
    trigger.add_argument("--branch", required=True)
    trigger.add_argument("--sha", required=True)
    trigger.add_argument("--source-run-id", required=True)
    trigger.add_argument("--source-run-attempt", required=True)
    trigger.add_argument("--source-workflow", required=True)
    trigger.add_argument("--scene-set", required=True, choices=["both", "public", "private"])
    trigger.add_argument("--publish", required=True)
    trigger.set_defaults(func=trigger_jenkins)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except FatalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
