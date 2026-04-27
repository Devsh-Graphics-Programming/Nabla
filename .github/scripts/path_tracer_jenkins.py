#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
import tarfile
import time
import urllib.parse
import zipfile
from pathlib import Path

from ci_common import (
    CiError,
    basic_headers,
    commit_status,
    github_headers,
    job_path,
    jget,
    jpost,
    json_request,
    multipart,
    output,
    parameter,
    wait_build,
    wait_queue,
)


REPO = "Devsh-Graphics-Programming/Nabla"
BRANCH = "ptCLI"
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_run(run, config, scene_set, publish):
    if run.get("head_branch") != BRANCH:
        raise CiError(f"Expected {BRANCH}, got {run.get('head_branch')}.")
    if run.get("name") != "Build" or run.get("conclusion") != "success":
        raise CiError(f"Build run {run.get('id')} is not green.")
    if not SHA_RE.match(str(run.get("head_sha") or "")):
        raise CiError("Build run does not expose a valid SHA.")
    return {
        "run_id": run["id"],
        "build_config": config,
        "scene_set": scene_set,
        "publish": str(publish).lower(),
        "branch": run["head_branch"],
        "sha": run["head_sha"],
        "run_attempt": run["run_attempt"],
        "workflow": run["name"],
    }


def resolve(args):
    headers = github_headers(args.github_token)
    if args.repository != REPO:
        raise CiError(f"Expected repository {REPO}.")
    if args.event_name != "push":
        run = json_request("GET", f"https://api.github.com/repos/{REPO}/actions/runs/{args.run_id}", headers)
        values = validate_run(run, args.build_config, args.scene_set, args.publish)
        output(values)
        return
    deadline = time.monotonic() + 6 * 60 * 60
    query = urllib.parse.urlencode({"branch": args.branch, "event": "push", "head_sha": args.sha, "per_page": 10})
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/build-nabla.yml/runs?{query}"
    while time.monotonic() < deadline:
        runs = json_request("GET", url, headers).get("workflow_runs", [])
        runs = [run for run in runs if run.get("name") == "Build" and run.get("head_sha") == args.sha]
        runs.sort(key=lambda run: int(run.get("run_attempt") or 0), reverse=True)
        if runs and runs[0].get("status") == "completed":
            values = validate_run(runs[0], "RelWithDebInfo", "both", "true")
            output(values)
            print(f"Resolved Build run {values['run_id']} on {BRANCH} at {values['sha']}.")
            return
        print(f"Waiting for Build workflow for {args.sha} to complete...")
        time.sleep(60)
    raise CiError(f"Timed out waiting for Build workflow for {args.sha}.")


def copy_contents(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def package(args):
    artifact = Path(args.artifact_dir)
    tar_path = next(artifact.glob("*-install.tar"), None)
    if not tar_path:
        raise CiError("Install artifact tar was not downloaded.")
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(artifact, filter="data")
        except TypeError:
            tar.extractall(artifact)
    root = artifact / "build-ct" / "install"
    root = root if args.build_config == "Release" else root / args.build_config.lower()
    runtime = root / "runtime"
    binary = root / "exe" / "examples_tests" / "40_PathTracer" / "bin"
    out = Path(args.package_root)
    zip_path = Path(args.zip_path)
    shutil.rmtree(out, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    prefix = Path() if args.build_config == "Release" else Path(args.build_config.lower())
    copy_contents(runtime, out / prefix / "runtime")
    copy_contents(binary, out / prefix / "exe" / "examples_tests" / "40_PathTracer" / "bin")
    if not list(out.rglob("40_pathtracer*.exe")) or not list(out.rglob("Nabla_*.dll")) or not list(out.rglob("dxcompiler.dll")):
        raise CiError("EX40 package is missing required runtime files.")
    if not (next(out.rglob("40_pathtracer*.exe")).parent / "report").exists():
        raise CiError("EX40 report template was not found in the package.")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(out).as_posix())
    if zip_path.stat().st_size > 200 * 1024 * 1024:
        raise CiError("EX40 package is unexpectedly large.")
    output({"zip_path": zip_path})
    print(f"Prepared EX40 runtime package: {zip_path.stat().st_size} bytes.")


def set_status(args, context, state, target, description):
    commit_status(REPO, args.sha, args.github_token, context, state, target, description)


def stale(actions, args):
    return parameter(actions, "SOURCE_REPOSITORY") == REPO and parameter(actions, "SOURCE_BRANCH") == BRANCH and not (
        parameter(actions, "SOURCE_RUN_ID") == args.source_run_id and parameter(actions, "SOURCE_RUN_ATTEMPT") == args.source_run_attempt
    )


def stop_old(base, headers, job, args):
    path = job_path(job)
    tree = urllib.parse.quote("builds[number,building,actions[parameters[name,value]]]", safe="")
    for build in jget(base, headers, f"/{path}/api/json?tree={tree}").get("builds", []):
        if build.get("building") and stale(build.get("actions"), args):
            jpost(base, headers, f"/{path}/{build['number']}/stop")
    tree = urllib.parse.quote("items[id,task[fullName],actions[parameters[name,value]]]", safe="")
    for item in jget(base, headers, f"/queue/api/json?tree={tree}").get("items", []):
        if (item.get("task") or {}).get("fullName") == job and stale(item.get("actions"), args):
            jpost(base, headers, f"/queue/cancelItem?id={item['id']}")


def start_job(args, headers, suite):
    job = f"ci/ditt/real/ex40-{suite}"
    context = f"jenkins/path-tracer-{suite}"
    source_url = f"https://github.com/{REPO}/actions/runs/{args.source_run_id}"
    set_status(args, context, "pending", source_url, f"Waiting for Jenkins {suite} path tracer run.")
    stop_old(args.jenkins_url, headers, job, args)
    content_type, body = multipart({
        "FAIL_ON_RENDER_FAILURE": "false",
        "PUBLISH": args.publish.lower(),
        "SOURCE_REPOSITORY": REPO,
        "SOURCE_BRANCH": BRANCH,
        "SOURCE_SHA": args.sha,
        "SOURCE_RUN_ID": args.source_run_id,
        "SOURCE_RUN_ATTEMPT": args.source_run_attempt,
        "SOURCE_WORKFLOW": args.source_workflow,
    }, "EX40_PACKAGE_FILE", args.package_path)
    _, response_headers, _ = jpost(args.jenkins_url, headers, f"/{job_path(job)}/buildWithParameters", body, content_type)
    number = wait_queue(args.jenkins_url, headers, response_headers["Location"], job)
    build_url = f"{args.jenkins_url.rstrip('/')}/{job_path(job)}/{number}/"
    set_status(args, context, "pending", build_url, f"Jenkins {suite} path tracer build #{number} is running.")
    print(f"Started Jenkins {job} #{number}: {build_url}")
    result = wait_build(args.jenkins_url, headers, job, number)
    if result not in {"SUCCESS", "UNSTABLE"}:
        set_status(args, context, "failure", build_url, f"Jenkins {suite} path tracer finished with {result}.")
        raise CiError(f"Jenkins {job} #{number} finished with {result}.")
    set_status(args, context, "success", build_url, f"Jenkins {suite} path tracer {result.lower()}.")
    return job, number, result, build_url


def trigger(args):
    if args.repository != REPO or args.branch != BRANCH or not SHA_RE.match(args.sha):
        raise CiError("Invalid source metadata.")
    if not Path(args.package_path).exists() or Path(args.package_path).stat().st_size > 200 * 1024 * 1024:
        raise CiError("Invalid package path or size.")
    headers = basic_headers(args.jenkins_user, args.jenkins_token)
    try:
        crumb = jget(args.jenkins_url, headers, "/crumbIssuer/api/json")
        headers[crumb["crumbRequestField"]] = crumb["crumb"]
    except Exception:
        pass
    suites = ["public", "private"] if args.scene_set == "both" else [args.scene_set]
    for result in [start_job(args, headers, suite) for suite in suites]:
        print(f"{result[0]} #{result[1]} {result[2]} {result[3]}")


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--github-token", required=True); r.add_argument("--repository", required=True); r.add_argument("--event-name", required=True)
    r.add_argument("--branch", default=""); r.add_argument("--sha", default=""); r.add_argument("--run-id", default="")
    r.add_argument("--build-config", default="RelWithDebInfo"); r.add_argument("--scene-set", default="both"); r.add_argument("--publish", default="true")
    r.set_defaults(func=resolve)
    k = sub.add_parser("package")
    k.add_argument("--artifact-dir", required=True); k.add_argument("--build-config", required=True); k.add_argument("--package-root", required=True); k.add_argument("--zip-path", required=True)
    k.set_defaults(func=package)
    t = sub.add_parser("trigger")
    for name in ["jenkins-url", "jenkins-user", "jenkins-token", "github-token", "package-path", "repository", "branch", "sha", "source-run-id", "source-run-attempt", "source-workflow", "scene-set", "publish"]:
        t.add_argument(f"--{name}", required=True)
    t.set_defaults(func=trigger)
    return p


if __name__ == "__main__":
    try:
        args = parser().parse_args()
        args.func(args)
    except CiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
