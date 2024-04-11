#ifndef _NBL_ASSET_S_PREPROCESSING_DEPENDENCY_H_INCLUDED_
#define _NBL_ASSET_S_PREPROCESSING_DEPENDENCY_H_INCLUDED_

#include <array>

namespace nbl::asset {
	struct SPreprocessingDependency
	{
		// Perf note: hashing while preprocessor lexing is likely to be slower than just hashing the whole array like this 
		inline SPreprocessingDependency(const system::path& _requestingSourceDir, const std::string_view& _identifier, const std::string_view& _contents) :
			requestingSourceDir(_requestingSourceDir), identifier(_identifier), contents(_contents)
		{
			assert(!_contents.empty());
			const auto reqDirStr = requestingSourceDir.make_preferred().string();
			std::vector<char> hashable;
			hashable.insert(hashable.end(), reqDirStr.data()[0], reqDirStr.data()[reqDirStr.size()]);
			hashable.insert(hashable.end(), identifier.data()[0], identifier.data()[identifier.size()]);
			hashable.insert(hashable.end(), _contents.data()[0], _contents.data()[_contents.size()]);
			// Can't static cast here?
			hash = nbl::core::XXHash_256((uint8_t*)(hashable.data()), hashable.size() * (sizeof(char) / sizeof(uint8_t)));
		}

		inline SPreprocessingDependency(SPreprocessingDependency&&) = default;
		inline SPreprocessingDependency& operator=(SPreprocessingDependency&&) = default;

		inline bool operator==(const SPreprocessingDependency& other) const
		{
			return hash == other.hash && identifier == identifier && contents == contents;
		}

		// path or identifier
		system::path requestingSourceDir = "";
		std::string identifier = "";
		// file contents
		std::string contents = "";
		// hash of the contents
		std::array<uint64_t, 4> hash = {};
		// If true, then `getIncludeStandard` was used to find, otherwise `getIncludeRelative`
		bool standardInclude = false;
		nbl::system::IFileBase::time_point_t lastWriteTime = {};

		SPreprocessingDependency() = default;
	};
}

#endif