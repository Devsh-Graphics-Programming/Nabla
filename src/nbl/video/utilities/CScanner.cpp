#include "nbl/video/utilities/CScanner.h"

using namespace nbl;
using namespace video;

//
IGPUSpecializedShader* CScanner::getDefaultSpecializedShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op)
{
	if (!m_specialized_shaders[scanType][dataType][op]) 
	{
		auto system = m_device->getPhysicalDevice()->getSystem();
		auto glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/scan/default.comp")>();
		auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl),asset::ICPUShader::buffer_contains_glsl);
		const char* storageType = nullptr;
		switch (dataType)
		{
			case EDT_UINT:
				storageType = "uint";
				break;
			case EDT_INT:
				storageType = "int";
				break;
			case EDT_FLOAT:
				storageType = "float";
				break;
			default:
				assert(false);
				break;
		}
		auto shader = m_device->createGPUShader(
			asset::IGLSLCompiler::createOverridenCopy(
				cpushader.get(),
				"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_GLSL_SCAN_TYPE_ %d\n#define _NBL_GLSL_SCAN_STORAGE_TYPE_ %s\n#define _NBL_GLSL_SCAN_BIN_OP_ %d\n",
				m_wg_size,uint32_t(scanType),storageType,uint32_t(op)
			)
		);
		m_specialized_shaders[scanType][dataType][op] = m_device->createGPUSpecializedShader(
			shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE,"nbl/builtin/glsl/scan/default.comp"}
		);
	}
	return m_specialized_shaders[scanType][dataType][op].get();
}