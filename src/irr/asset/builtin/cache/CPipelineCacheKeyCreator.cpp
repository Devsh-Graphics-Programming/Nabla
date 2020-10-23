#include "CPipelineCacheKeyCreator.h"

namespace irr
{
	namespace asset
	{
		bool CPipelineCacheKeyCreator<ICPURenderpassIndependentPipeline>::createCacheKey(core::smart_refctd_ptr<IAsset> asset)
		{
			auto pipeline = core::smart_refctd_ptr_dynamic_cast<ICPURenderpassIndependentPipeline>(asset);
			if (pipeline.get())
			{
				cacheKey.clear();

				const SVertexInputParams& vertexInputParams = pipeline->getVertexInputParams();
				const SPrimitiveAssemblyParams& primitiveAssemblyParams = pipeline->getPrimitiveAssemblyParams();
				const SBlendParams& blendParams = pipeline->getBlendParams();
				const SRasterizationParams& rasterizationParams = pipeline->getRasterizationParams();

				const auto& [enabledAttribFlags, enabledBindingFlags, attributes, bindings] = vertexInputParams;
				{
					cacheKey += ICacheKeyCreator::getNewCommmaValue(enabledAttribFlags);
					cacheKey += ICacheKeyCreator::getNewCommmaValue(enabledBindingFlags);

					for (size_t i = 0; i < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++i)
					{
						const auto& attribute = attributes[i];

						cacheKey += ICacheKeyCreator::getNewCommmaValue(attribute.binding);
						cacheKey += ICacheKeyCreator::getNewCommmaValue(attribute.format);
						cacheKey += ICacheKeyCreator::getNewCommmaValue(attribute.relativeOffset);
					}

					for (size_t i = 0; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
					{
						const auto& binding = bindings[i];

						cacheKey += ICacheKeyCreator::getNewCommmaValue(binding.inputRate);
						cacheKey += ICacheKeyCreator::getNewCommmaValue(binding.stride);
					}
				}

				// TODO
				// actually I don't think it will be good to translate all to string cache key
			}
			else
				return false;
		}
		
		bool CPipelineCacheKeyCreator<ICPUComputePipeline>::createCacheKey(core::smart_refctd_ptr<IAsset> asset)
		{
			auto pipeline = core::smart_refctd_ptr_dynamic_cast<ICPUComputePipeline>(asset);
			if (pipeline.get())
			{
				return false; // TODO
			}
			else
				return false;
		}
	}
}