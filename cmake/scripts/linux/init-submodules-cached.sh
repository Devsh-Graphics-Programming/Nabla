#!/usr/bin/env bash
# Hydrate the submodules of a Nabla checkout from an already hydrated local
# clone (the "cache") instead of downloading them again from GitHub.
#
# Usage: init-submodules-cached.sh <cache-checkout> [<target-checkout>]
#
# For every submodule, recursively, the URL is overridden in the target's local
# git config (never in .gitmodules) to point at the matching submodule work
# tree inside the cache, then `git submodule update --init` clones it. A local
# clone hard-links objects, so the target does not depend on the cache's object
# store afterwards (no alternates, safe if the cache is ever repacked).
#
# Submodules that are not hydrated in the cache are skipped: these are the ones
# excluded by policy (`update = none`, private repositories). Run with
# NBL_UPDATE_GIT_SUBMODULE=OFF afterwards so CMake does not try to update them.

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "usage: $0 <cache-checkout> [<target-checkout>]" >&2
	exit 2
fi

CACHE_ROOT="$(cd "$1" && pwd)"
TARGET_ROOT="$(cd "${2:-.}" && git rev-parse --show-toplevel)"

rel() { [[ "$1" == "$TARGET_ROOT" ]] && echo "" || echo "${1#$TARGET_ROOT/}/"; }

hydrate() {
	local target="$1" cache="$2"
	[[ -f "$target/.gitmodules" ]] || return 0

	local key name path
	while read -r key path; do
		name="${key#submodule.}"
		name="${name%.path}"

		if [[ ! -e "$cache/$path/.git" ]] || ! git -C "$cache/$path" rev-parse --verify -q HEAD >/dev/null; then
			echo "skip  $(rel "$target")$path (not hydrated in cache)"
			continue
		fi
		if [[ "$(git -C "$cache/$path" rev-parse --show-toplevel)" != "$cache/$path" ]]; then
			echo "skip  $(rel "$target")$path (cache path is not a checkout)"
			continue
		fi

		echo "init  $(rel "$target")$path"
		git -C "$target" config "submodule.$name.url" "$cache/$path"
		git -C "$target" -c protocol.file.allow=always submodule update --init -- "$path"

		# carry over cache-local attribute fixes (e.g. line-ending overrides)
		local cache_attr target_attr
		cache_attr="$(git -C "$cache/$path" rev-parse --git-path info/attributes)"
		target_attr="$(git -C "$target/$path" rev-parse --git-path info/attributes)"
		[[ "$cache_attr" = /* ]] || cache_attr="$cache/$path/$cache_attr"
		[[ "$target_attr" = /* ]] || target_attr="$target/$path/$target_attr"
		if [[ -s "$cache_attr" ]]; then
			mkdir -p "$(dirname "$target_attr")"
			cp "$cache_attr" "$target_attr"
		fi

		hydrate "$target/$path" "$cache/$path"
	done < <(git -C "$target" config -f .gitmodules --get-regexp '^submodule\..*\.path$' || true)
}

hydrate "$TARGET_ROOT" "$CACHE_ROOT"

# A matching SHA is not proof of a populated work tree (see NAB-5): verify.
dirty="$(git -C "$TARGET_ROOT" submodule foreach --quiet --recursive 'git status --porcelain' | wc -l)"
if [[ "$dirty" != "0" ]]; then
	echo "error: $dirty dirty entries in submodule work trees" >&2
	exit 1
fi
echo "ok: submodules hydrated and clean"
