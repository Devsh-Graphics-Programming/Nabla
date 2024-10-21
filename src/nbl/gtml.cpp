#include "git_info.h"

namespace nbl {
	const gtml::GitInfo& getGitInfo(gtml::E_GIT_REPO_META repo) {
		return gtml::gitMeta[repo];
	}
}