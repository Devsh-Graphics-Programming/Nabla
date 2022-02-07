#include "nbl/video/utilities/CScanner.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<asset::ICPUShader> CScanner::createShader(const bool indirect, const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const
{
    auto system = m_device->getPhysicalDevice()->getSystem();
    core::smart_refctd_ptr<nbl::system::IFile> glsl;
    {
        if(indirect)
            glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/scan/indirect.comp")>();
        else
            glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/scan/direct.comp")>();
    }
    auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glsl->getSize());
    memcpy(buffer->getPointer(), glsl->getMappedPointer(), glsl->getSize());
    auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "????");
    const char* storageType = nullptr;
    switch(dataType)
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

    return asset::IGLSLCompiler::createOverridenCopy(
        cpushader.get(),
        "#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_GLSL_WORKGROUP_SIZE_LOG2_ %d\n#define _NBL_GLSL_SCAN_TYPE_ %d\n#define _NBL_GLSL_SCAN_STORAGE_TYPE_ %s\n#define _NBL_GLSL_SCAN_BIN_OP_ %d\n",
        m_workgroupSize, core::findMSB(m_workgroupSize), uint32_t(scanType), storageType, uint32_t(op));
}