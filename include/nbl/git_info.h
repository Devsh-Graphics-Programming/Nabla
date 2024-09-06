#ifndef __GIT_INFO_H_INCLUDED__
#define __GIT_INFO_H_INCLUDED__

namespace nbl {
	struct GitInfo {
		const char* CommitHash;
		const char* ShortCommitHash;
	};
	extern const GitInfo git;
}

#endif // __GIT_INFO_H_INCLUDED__