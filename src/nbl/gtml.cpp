#include "nbl/git/info.h"

namespace nbl {
	const ::gtml::IGitInfo& getGitInfo(gtml::E_GIT_REPO_META repo) {
		return *gtml::gitMeta[repo];
	}
}
