#include "nbl/video/utilities/CScanner.h"

using namespace nbl;
using namespace video;

#if 0 // TODO: port
core::smart_refctd_ptr<asset::ICPUShader> CScanner::createShader(const bool indirect, const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const
{
	auto system = m_device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<const system::IFile> glsl;
	{
		auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
		{
			nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
			system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
			if (future.wait())
				return future.copy();
			return nullptr;
		};

		if(indirect)
			glsl = loadBuiltinData("nbl/builtin/glsl/scan/indirect.comp");
		else
			glsl = loadBuiltinData("nbl/builtin/glsl/scan/direct.comp");
	}
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glsl->getSize());
	memcpy(buffer->getPointer(), glsl->getMappedPointer(), glsl->getSize());
	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "????");
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

	return asset::CGLSLCompiler::createOverridenCopy(
		cpushader.get(),
		"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_GLSL_WORKGROUP_SIZE_LOG2_ %d\n#define _NBL_GLSL_SCAN_TYPE_ %d\n#define _NBL_GLSL_SCAN_STORAGE_TYPE_ %s\n#define _NBL_GLSL_SCAN_BIN_OP_ %d\n",
		m_workgroupSize,hlsl::findMSB(m_workgroupSize),uint32_t(scanType),storageType,uint32_t(op)
	);
}
#endif