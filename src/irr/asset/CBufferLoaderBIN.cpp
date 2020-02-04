#include "CBufferLoaderBIN.h"

namespace irr
{
	namespace asset
	{
		asset::SAssetBundle CBufferLoaderBIN::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			SContext ctx(_file->getSize());
			ctx.file = _file;

			ctx.file->read(ctx.sourceCodeBuffer.get()->getPointer(), ctx.file->getSize());

			return SAssetBundle({std::move(ctx.sourceCodeBuffer)});
		}

		bool CBufferLoaderBIN::isALoadableFileFormat(io::IReadFile* _file) const
		{
			return true; // validation if needed
		}
	}
}