
#include "nabla.h"
#include "TextRendering.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

namespace nbl
{
namespace ext
{
namespace TextRendering
{

TextRenderer::TextRenderer(core::smart_refctd_ptr<ILogicalDevice>&& device, uint32_t maxGlyphCount, uint32_t maxStringCount, uint32_t maxGlyphsPerString):
	m_device(std::move(device))
{
	m_geomDataBuffer = core::make_smart_refctd_ptr<glyph_geometry_pool_t>(device.get());
	m_stringDataPropertyPool = core::make_smart_refctd_ptr<string_pool_t>(device.get());

	{
		// Global string DS
		const uint32_t bindingCount = 2u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
		{
			// Geometry data buffer SSBO
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_STORAGE_IMAGE;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE | asset::IShader::ESS_VERTEX;
			bindings[0].samplers = nullptr;

			// String data property pool SSBO
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE | asset::IShader::ESS_VERTEX;
			bindings[1].samplers = nullptr;
		}
		m_globalStringDSLayout =
			logicalDevice->createDescriptorSetLayout(bindings, bindings + bindingCount);

		const uint32_t descriptorPoolSizeCount = 1u;
		video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
		poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
		poolSizes[0].count = 2u;

		video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= logicalDevice->createDescriptorPool(descriptorPoolFlags, 1,
				descriptorPoolSizeCount, poolSizes);

		m_globalStringDS = logicalDevice->createDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(m_globalStringDSLayout));
	}

	template<class Clock = typename std::chrono::steady_clock>
	uint32_t TextRenderer::allocateStrings(
		const std::chrono::time_point<Clock>&maxWaitPoint, // lets have also a version without this (default)
		const uint32_t count, // how many strings
		string_handle_t* handles, // output handles, if `glyphDataAddr` was not primed with invalid_address, allocation will not happen, likewise for `stringDataAddr`
		const char* const* stringData,
		const StringBoundingBox* wrappingBoxes = nullptr // optional, to wrap paragraphs
	)
	{
		for (char* cur = *stringData; *cur != '\0'; cur++)
		{
			// lookup where each character is in the atlas
			// generate some glyphs?
		}
	}

}

}
}
}
