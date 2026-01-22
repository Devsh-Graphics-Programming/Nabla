import argparse
import concurrent.futures
import json
import random
import re
import statistics
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake")
    parser.add_argument("--build-dir")
    parser.add_argument("--target")
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--depfile", default="")
    parser.add_argument("--shader-cache", default="")
    parser.add_argument("--preprocess-cache", default="")
    parser.add_argument("--preprocessed", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=3)
    parser.add_argument("--budget-ms", type=int, default=0)
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def normalize_command(cmd):
    if not cmd:
        return []
    return [arg for arg in cmd if arg]


def strip_option(cmd, flag, takes_value):
    out = []
    skip = False
    for arg in cmd:
        if skip:
            skip = False
            continue
        if arg == flag:
            if takes_value:
                skip = True
            continue
        out.append(arg)
    return out


def strip_options(cmd, options_with_values, options_flags):
    result = cmd
    for flag in options_with_values:
        result = strip_option(result, flag, True)
    for flag in options_flags:
        result = strip_option(result, flag, False)
    return result


def replace_option_value(cmd, flag, value):
    result = list(cmd)
    for idx in range(len(result) - 1):
        if result[idx] == flag:
            result[idx + 1] = value
    return result


def apply_output_overrides(cmd, args):
    result = list(cmd)
    if args.output:
        result = replace_option_value(result, "-Fc", args.output)
    if args.depfile:
        result = replace_option_value(result, "-MF", args.depfile)
    if args.report:
        result = replace_option_value(result, "-nbl-report", args.report)
    if args.log:
        result = replace_option_value(result, "-log", args.log)
    return result


def command_without_shader_cache(cmd):
    return strip_options(
        cmd,
        options_with_values=["-shader-cache-file", "-nbl-shader-cache-compression", "-shader-cache-compression"],
        options_flags=["-nbl-shader-cache", "-shader-cache"],
    )


def command_without_preprocess_cache(cmd):
    return strip_options(
        cmd,
        options_with_values=["-preprocess-cache-file"],
        options_flags=["-nbl-preprocess-cache", "-preprocess-cache"],
    )


def command_without_preamble(cmd):
    return strip_options(cmd, options_with_values=[], options_flags=["-nbl-preprocess-preamble"])

def command_without_all_caches(cmd):
    cmd = command_without_shader_cache(cmd)
    cmd = command_without_preprocess_cache(cmd)
    cmd = command_without_preamble(cmd)
    return cmd


def run_cmd(args):
    cmd = normalize_command(args)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError("command failed")


def run_build(args, command_override=None):
    if command_override is not None:
        run_cmd(command_override)
        return
    if args.command:
        run_cmd(args.command)
        return
    if not (args.cmake and args.build_dir and args.target):
        raise RuntimeError("missing --command or --cmake/--build-dir/--target")
    cmd = [args.cmake, "--build", args.build_dir, "--target", args.target]
    if args.config:
        cmd.extend(["--config", args.config])
    run_cmd(cmd)


def load_report(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_body_value(path, value):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"uint\s+sink\s*=\s*(\d+)u;", text)
    if not match:
        raise RuntimeError("missing body marker: uint sink = <n>u;")
    current = int(match.group(1))
    if current == value:
        return
    replacement = f"uint sink = {value}u;"
    new_text = re.sub(r"uint\s+sink\s*=\s*\d+u;", replacement, text, count=1)
    path.write_text(new_text, encoding="utf-8")


def pick_body_value(rng, exclude):
    choices = [2, 3, 5, 7]
    value = rng.choice(choices)
    if value == exclude:
        value = choices[(choices.index(value) + 1) % len(choices)]
    return value


def pick_defines(rng):
    pool = [
        "#define NBL_NSC_TEST_DEF_A 1",
        "#define NBL_NSC_TEST_DEF_B 2",
        "#define NBL_NSC_TEST_DEF_C 3",
        "#define NBL_NSC_TEST_DEF_D 4",
    ]
    rng.shuffle(pool)
    count = rng.randint(1, 3)
    return pool[:count]


def normalized_includes():
    return [
        "#include <nbl/builtin/hlsl/cpp_compat/./intrinsics.hlsl>",
        "#include <nbl/builtin/hlsl/cpp_compat/../cpp_compat/matrix.hlsl>",
        "#include <nbl\\builtin\\hlsl\\cpp_compat\\vector.hlsl>",
    ]


def default_builtin_includes():
    return [
        "#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>",
        "#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>",
        "#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>",
    ]


def replace_section(text, begin, end, lines):
    begin_idx = text.find(begin)
    end_idx = text.find(end, begin_idx)
    if begin_idx == -1 or end_idx == -1:
        raise RuntimeError(f"missing proxy markers: {begin} / {end}")
    end_idx += len(end)
    content = "\n".join(lines)
    if content:
        content = f"\n{content}\n"
    else:
        content = "\n"
    return text[:begin_idx] + begin + content + end + text[end_idx:]


def set_proxy_defines(proxy_path, defines):
    text = proxy_path.read_text(encoding="utf-8")
    updated = replace_section(
        text,
        "// NBL_NSC_CACHE_TEST_DEFINES_BEGIN",
        "// NBL_NSC_CACHE_TEST_DEFINES_END",
        defines,
    )
    proxy_path.write_text(updated, encoding="utf-8")


def set_proxy_includes(proxy_path, includes):
    text = proxy_path.read_text(encoding="utf-8")
    updated = replace_section(
        text,
        "// NBL_NSC_CACHE_TEST_INCLUDES_BEGIN",
        "// NBL_NSC_CACHE_TEST_INCLUDES_END",
        includes,
    )
    proxy_path.write_text(updated, encoding="utf-8")


def delete_path(path):
    if path.exists():
        path.unlink()


def assert_true(expr, message):
    if not expr:
        raise RuntimeError(message)


def assert_not_exists(path, message):
    if path.exists():
        raise RuntimeError(message)


def assert_eq(actual, expected, message):
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected}, got {actual}")

def cleanup_outputs(output_path, report_path, args):
    delete_path(output_path)
    delete_path(report_path)

    log_path = Path(args.log) if args.log else Path(str(output_path) + ".log")
    depfile_path = Path(args.depfile) if args.depfile else Path(str(output_path) + ".dep")
    shader_cache_path = Path(args.shader_cache) if args.shader_cache else Path(str(output_path) + ".ppcache")
    preprocess_cache_path = Path(args.preprocess_cache) if args.preprocess_cache else Path(str(output_path) + ".ppcache.pre")
    preprocessed_path = Path(args.preprocessed) if args.preprocessed else Path(str(output_path) + ".pre.hlsl")

    delete_path(log_path)
    delete_path(depfile_path)
    delete_path(shader_cache_path)
    delete_path(preprocess_cache_path)
    delete_path(preprocessed_path)


def assert_exists(path, message):
    if not path.exists():
        raise RuntimeError(message)


def check_artifacts(output_path, report_path, args, expect_shader_cache=True, expect_preprocess_cache=True, expect_preprocessed=True):
    log_path = Path(args.log) if args.log else Path(str(output_path) + ".log")
    depfile_path = Path(args.depfile) if args.depfile else Path(str(output_path) + ".dep")
    shader_cache_path = Path(args.shader_cache) if args.shader_cache else Path(str(output_path) + ".ppcache")
    preprocess_cache_path = Path(args.preprocess_cache) if args.preprocess_cache else Path(str(output_path) + ".ppcache.pre")
    preprocessed_path = Path(args.preprocessed) if args.preprocessed else Path(str(output_path) + ".pre.hlsl")

    assert_exists(output_path, "output .spv not found after cold run")
    assert_exists(report_path, "report not found after cold run")
    assert_exists(log_path, "log not found after cold run")
    assert_exists(depfile_path, "depfile not found after cold run")
    if expect_shader_cache:
        assert_exists(shader_cache_path, "shader cache not found after cold run")
    else:
        assert_not_exists(shader_cache_path, "shader cache should not be created")
    if expect_preprocess_cache:
        assert_exists(preprocess_cache_path, "preprocess cache not found after cold run")
    else:
        assert_not_exists(preprocess_cache_path, "preprocess cache should not be created")
    if expect_preprocessed:
        assert_exists(preprocessed_path, "preprocessed output not found after cold run")
    else:
        assert_not_exists(preprocessed_path, "preprocessed output should not be created")


def normalize_dep_path(text):
    return text.replace("\\", "/")


def assert_report_schema(report):
    assert_true(isinstance(report, dict), "report should be an object")
    required_sections = ["shader_cache", "preprocess_cache", "compile", "output", "input", "total_ms"]
    for key in required_sections:
        assert_true(key in report, f"report missing key: {key}")
    assert_true(isinstance(report["shader_cache"], dict), "shader_cache should be object")
    assert_true(isinstance(report["preprocess_cache"], dict), "preprocess_cache should be object")
    assert_true(isinstance(report["compile"], dict), "compile should be object")
    assert_true(isinstance(report["output"], dict), "output should be object")
    assert_true(isinstance(report["input"], str), "input should be string")
    assert_true(isinstance(report["total_ms"], int), "total_ms should be int")
    if "preamble" in report:
        assert_true(isinstance(report["preamble"], dict), "preamble should be object")


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def report_time_ms(report):
    return int(report.get("total_with_output_ms", report.get("total_ms", 0)))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    proxy_path = input_path.parent / "proxy.hlsl"

    if args.mode == "shader_cache_cold":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        check_artifacts(output_path, report_path, args)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected on cold run")
        assert_eq(report["preprocess_cache"]["status"], "miss", "preprocess cache miss expected on cold run")
        assert_eq(report["compile"]["called"], True, "compile should run on cold run")
        return

    if args.mode == "shader_cache_hit":
        set_body_value(input_path, 1)
        delete_path(output_path)
        delete_path(report_path)
        run_build(args)
        assert_true(report_path.exists(), "report not found")
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], True, "shader cache hit expected")
        assert_eq(report["preprocess_cache"]["status"], "skipped", "preprocess cache should be skipped on shader hit")
        assert_eq(report["compile"]["called"], False, "compile should be skipped on shader cache hit")
        return

    if args.mode == "preprocess_cache_cold":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        check_artifacts(output_path, report_path, args)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected on cold run")
        assert_eq(report["preprocess_cache"]["status"], "miss", "preprocess cache miss expected on cold run")
        assert_eq(report["compile"]["called"], True, "compile should run on cold run")
        return

    if args.mode == "preprocess_cache_hit":
        set_body_value(input_path, 1)
        set_body_value(input_path, pick_body_value(rng, 1))
        delete_path(report_path)
        run_build(args)
        assert_true(report_path.exists(), "report not found")
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache should miss on body change")
        assert_eq(report["preprocess_cache"]["status"], "hit", "preprocess cache hit expected")
        assert_eq(report["compile"]["called"], True, "compile should run on preprocess cache hit")
        assert_true(report["preamble"]["used"] is False, "preamble should be unused")
        set_body_value(input_path, 1)
        return

    if args.mode == "preamble_cache_cold":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        check_artifacts(output_path, report_path, args)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected on cold run")
        assert_eq(report["preprocess_cache"]["status"], "miss", "preprocess cache miss expected on cold run")
        assert_eq(report["compile"]["called"], True, "compile should run on cold run")
        return

    if args.mode == "preamble_cache_hit":
        set_body_value(input_path, 1)
        set_body_value(input_path, pick_body_value(rng, 1))
        delete_path(report_path)
        run_build(args)
        assert_true(report_path.exists(), "report not found")
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache should miss on body change")
        assert_eq(report["preprocess_cache"]["status"], "hit", "preprocess cache hit expected")
        assert_eq(report["compile"]["called"], True, "compile should run on preamble hit")
        assert_true(report["preamble"]["used"] is True, "preamble should be used")
        set_body_value(input_path, 1)
        return

    if args.mode == "preamble_cache_hit_time":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        set_body_value(input_path, pick_body_value(rng, 1))
        delete_path(report_path)
        run_build(args)
        assert_true(report_path.exists(), "report not found")
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache should miss on body change")
        assert_eq(report["preprocess_cache"]["status"], "hit", "preprocess cache hit expected")
        assert_eq(report["compile"]["called"], True, "compile should run on preamble hit")
        assert_true(report["preamble"]["used"] is True, "preamble should be used")
        total_ms = report_time_ms(report)
        print(f"preamble_hit_total_with_output_ms={total_ms}")
        if args.budget_ms > 0:
            assert_true(total_ms <= args.budget_ms, "preamble hit time budget exceeded")
        set_body_value(input_path, 1)
        return

    if args.mode == "no_cache_cold":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        cmd = apply_output_overrides(command_without_all_caches(normalize_command(args.command)), args)
        run_build(args, cmd)
        check_artifacts(output_path, report_path, args, expect_shader_cache=False, expect_preprocess_cache=False, expect_preprocessed=False)
        report = load_report(report_path)
        assert_true(report.get("shader_cache", {}).get("enabled") is False, "shader cache should be disabled")
        assert_true(report.get("preprocess_cache", {}).get("enabled") is False, "preprocess cache should be disabled")
        preamble = report.get("preamble", {})
        assert_true(preamble.get("enabled") is False, "preamble should be disabled")
        assert_eq(report["compile"]["called"], True, "compile should run with caches disabled")
        return

    if args.mode == "shader_cache_disabled":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        cmd = command_without_shader_cache(normalize_command(args.command))
        run_build(args, cmd)
        check_artifacts(output_path, report_path, args, expect_shader_cache=False, expect_preprocess_cache=True, expect_preprocessed=True)
        report = load_report(report_path)
        assert_true(report.get("shader_cache", {}).get("enabled") is False, "shader cache should be disabled")
        assert_eq(report["compile"]["called"], True, "compile should run when shader cache is disabled")
        return

    if args.mode == "preprocess_cache_disabled":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        cmd = command_without_preprocess_cache(normalize_command(args.command))
        run_build(args, cmd)
        check_artifacts(output_path, report_path, args, expect_shader_cache=True, expect_preprocess_cache=False, expect_preprocessed=False)
        report = load_report(report_path)
        assert_true(report.get("preprocess_cache", {}).get("enabled") is False, "preprocess cache should be disabled")
        assert_eq(report["compile"]["called"], True, "compile should run when preprocess cache is disabled")
        return

    if args.mode == "preamble_cache_disabled":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        cmd = command_without_preamble(normalize_command(args.command))
        run_build(args, cmd)
        set_body_value(input_path, pick_body_value(rng, 1))
        delete_path(report_path)
        run_build(args, cmd)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache should miss on body change")
        assert_eq(report["preprocess_cache"]["status"], "hit", "preprocess cache hit expected")
        preamble = report.get("preamble", {})
        assert_true(preamble.get("enabled") is False, "preamble should be disabled")
        assert_true(preamble.get("used", False) is False, "preamble should not be used when disabled")
        set_body_value(input_path, 1)
        return

    if args.mode == "shader_cache_isolation":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        cmd = command_without_preprocess_cache(command_without_preamble(normalize_command(args.command)))
        run_build(args, cmd)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected on first run")
        set_body_value(input_path, pick_body_value(rng, 1))
        delete_path(report_path)
        run_build(args, cmd)
        report = load_report(report_path)
        assert_eq(report["shader_cache"]["hit"], False, "shader cache should not hit on changed body")
        assert_eq(report["compile"]["called"], True, "compile should run on shader cache miss")
        set_body_value(input_path, 1)
        return

    if args.mode == "deps_invalidation":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        original_proxy = proxy_path.read_text(encoding="utf-8")
        try:
            run_build(args)
            report = load_report(report_path)
            assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected on cold run")
            assert_eq(report["preprocess_cache"]["status"], "miss", "preprocess cache miss expected on cold run")
            set_proxy_defines(proxy_path, pick_defines(rng))
            delete_path(report_path)
            run_build(args)
            report = load_report(report_path)
            assert_eq(report["shader_cache"]["hit"], False, "shader cache miss expected after dep change")
            assert_eq(report["preprocess_cache"]["status"], "miss", "preprocess cache miss expected after dep change")
        finally:
            proxy_path.write_text(original_proxy, encoding="utf-8")
        return

    if args.mode == "path_normalization":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        original_proxy = proxy_path.read_text(encoding="utf-8")
        try:
            set_proxy_includes(proxy_path, normalized_includes())
            run_build(args)
            delete_path(report_path)
            run_build(args)
            report = load_report(report_path)
            assert_eq(report["shader_cache"]["hit"], True, "shader cache hit expected with normalized includes")
            assert_eq(report["preprocess_cache"]["status"], "skipped", "preprocess cache should be skipped on shader hit")
        finally:
            proxy_path.write_text(original_proxy, encoding="utf-8")
        return

    if args.mode == "random_defines":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        original_proxy = proxy_path.read_text(encoding="utf-8")
        try:
            set_proxy_defines(proxy_path, pick_defines(rng))
            run_build(args)
            delete_path(report_path)
            run_build(args)
            report = load_report(report_path)
            assert_eq(report["shader_cache"]["hit"], True, "shader cache hit expected after randomized defines")
        finally:
            proxy_path.write_text(original_proxy, encoding="utf-8")
        return

    if args.mode == "parallel_smoke":
        base_cmd = normalize_command(args.command)
        base_output = output_path

        def make_paths(idx):
            new_output = base_output.with_name(f"{base_output.stem}.p{idx}{base_output.suffix}")
            new_report = Path(str(new_output) + ".report.json")
            new_log = Path(str(new_output) + ".log")
            new_dep = Path(str(new_output) + ".dep")
            return new_output, new_report, new_log, new_dep

        def worker(idx):
            new_output, new_report, new_log, new_dep = make_paths(idx)
            for p in [new_output, new_report, new_log, new_dep]:
                delete_path(p)
            cmd = replace_option_value(base_cmd, "-Fc", str(new_output))
            cmd = replace_option_value(cmd, "-MF", str(new_dep))
            cmd = replace_option_value(cmd, "-nbl-report", str(new_report))
            run_build(args, cmd)
            assert_true(new_report.exists(), "parallel report not found")
            load_report(new_report)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as pool:
            futures = [pool.submit(worker, idx) for idx in range(args.parallel_jobs)]
            for fut in futures:
                fut.result()
        return

    if args.mode == "stress":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        totals = []
        totals_with_output = []
        for _ in range(args.iterations):
            delete_path(report_path)
            run_build(args)
            report = load_report(report_path)
            totals.append(report_time_ms(report))
            totals_with_output.append(int(report.get("total_with_output_ms", report_time_ms(report))))
        if totals:
            print(f"stress total_ms median={statistics.median(totals)} p95={percentile(totals, 95)} samples={len(totals)}")
            print(f"stress total_with_output_ms median={statistics.median(totals_with_output)} p95={percentile(totals_with_output, 95)} samples={len(totals_with_output)}")
        return

    if args.mode == "report_schema":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        report = load_report(report_path)
        assert_report_schema(report)
        return

    if args.mode == "depfile_contents":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        run_build(args)
        depfile_path = Path(args.depfile) if args.depfile else Path(str(output_path) + ".dep")
        dep_text = normalize_dep_path(depfile_path.read_text(encoding="utf-8"))
        input_text = normalize_dep_path(str(input_path))
        proxy_text = normalize_dep_path(str(proxy_path))
        assert_true(input_text in dep_text, "depfile missing input path")
        assert_true(proxy_text in dep_text, "depfile missing proxy path")
        return

    if args.mode == "cache_path_override":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        override_shader = Path(str(output_path) + ".override.ppcache")
        override_preprocess = Path(str(output_path) + ".override.ppcache.pre")
        delete_path(override_shader)
        delete_path(override_preprocess)
        cmd = normalize_command(args.command)
        if not cmd:
            raise RuntimeError("missing command for cache override test")
        insert_at = max(len(cmd) - 1, 0)
        cmd = (
            cmd[:insert_at]
            + ["-shader-cache-file", str(override_shader), "-preprocess-cache-file", str(override_preprocess)]
            + cmd[insert_at:]
        )
        run_build(args, cmd)
        assert_exists(override_shader, "shader cache override file not created")
        assert_exists(override_preprocess, "preprocess cache override file not created")
        return

    if args.mode == "large_include_graph":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        original_proxy = proxy_path.read_text(encoding="utf-8")
        created = []
        try:
            for idx in range(25):
                inc_path = proxy_path.parent / f"dummy_inc_{idx}.hlsl"
                inc_path.write_text(f"// dummy {idx}\n", encoding="utf-8")
                created.append(inc_path)
            includes = default_builtin_includes()
            includes.extend([f"#include \"{p.name}\"" for p in created])
            set_proxy_includes(proxy_path, includes)
            run_build(args)
            depfile_path = Path(args.depfile) if args.depfile else Path(str(output_path) + ".dep")
            dep_text = normalize_dep_path(depfile_path.read_text(encoding="utf-8"))
            for idx in [0, len(created) // 2, len(created) - 1]:
                check_path = normalize_dep_path(str(created[idx]))
                assert_true(check_path in dep_text, "depfile missing dummy include")
        finally:
            proxy_path.write_text(original_proxy, encoding="utf-8")
            for p in created:
                delete_path(p)
        return

    if args.mode == "unused_include":
        set_body_value(input_path, 1)
        cleanup_outputs(output_path, report_path, args)
        original_proxy = proxy_path.read_text(encoding="utf-8")
        unused_path = proxy_path.parent / "unused_inc.hlsl"
        try:
            unused_path.write_text("// unused\n", encoding="utf-8")
            includes = default_builtin_includes()
            includes.extend(["#if 0", f"#include \"{unused_path.name}\"", "#endif"])
            set_proxy_includes(proxy_path, includes)
            run_build(args)
            depfile_path = Path(args.depfile) if args.depfile else Path(str(output_path) + ".dep")
            dep_text = normalize_dep_path(depfile_path.read_text(encoding="utf-8"))
            assert_true(normalize_dep_path(str(unused_path)) not in dep_text, "depfile should not include unused include")
        finally:
            proxy_path.write_text(original_proxy, encoding="utf-8")
            delete_path(unused_path)
        return

    raise RuntimeError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
