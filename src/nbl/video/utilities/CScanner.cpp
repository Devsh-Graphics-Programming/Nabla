#include "nbl/video/utilities/CScanner.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<asset::ICPUShader> CScanner::createShader(const bool indirect, const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const
{
	auto system = m_device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<const system::IFile> hlsl;
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
			hlsl = loadBuiltinData("nbl/builtin/hlsl/scan/indirect.hlsl");
		else
			hlsl = loadBuiltinData("nbl/builtin/hlsl/scan/direct.hlsl");
	}
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(hlsl->getSize());
	memcpy(buffer->getPointer(), hlsl->getMappedPointer(), hlsl->getSize());
	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_HLSL, "????");

	// (REVIEW): All of the below, that rely on enumerations, should probably be changed to take advantage of HLSL-CPP compatibility.
	// Issue is that CScanner caches all shaders for all permutations of scanType-dataType-operator and it's not clear how to 
	// do this without enums
	const char* storageType = nullptr;
	switch (dataType)
	{
		case EDT_UINT:
			storageType = "uint32_t";
			break;
		case EDT_INT:
			storageType = "int32_t";
			break;
		case EDT_FLOAT:
			storageType = "float32_t";
			break;
		default:
			assert(false);
			break;
	}

	bool isExclusive = scanType == EST_EXCLUSIVE;

	const char* binop = nullptr;
	switch (op)
	{
		case EO_AND:
			binop = "bit_and";
			break;
		case EO_OR:
			binop = "bit_or";
			break;
		case EO_XOR:
			binop = "bit_xor";
			break;
		case EO_ADD:
			binop = "plus";
			break;
		case EO_MUL:
			binop = "multiplies";
			break;
		case EO_MAX:
			binop = "maximum";
			break;
		case EO_MIN:
			binop = "minimum";
			break;
		default:
			assert(false);
			break;
	}

	return asset::CGLSLCompiler::createOverridenCopy(
		cpushader.get(),
		"#define WORKGROUP_SIZE %d\nconst bool isExclusive = %b\ntypedef storageType %s\n#define BINOP nbl::hlsl::%s\n",
		m_workgroupSize,isExclusive,storageType,binop
	);
}