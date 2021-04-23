// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include <algorithm>

#include "nbl/asset/utils/spvUtils.h"

#include "COpenGLSpecializedShader.h"
#include "COpenGLDriver.h"

#include "spirv_cross/spirv_parser.hpp"


#ifdef _NBL_COMPILE_WITH_OPENGL_

namespace nbl
{
namespace video
{

namespace impl
{
static void specialize(spirv_cross::CompilerGLSL& _comp, const asset::ISpecializedShader::SInfo& _specData)
{
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    for (const auto& sc : sconsts)
    {
        spirv_cross::SPIRConstant& val = _comp.get_constant(sc.id);

        auto specVal = _specData.getSpecializationByteValue(sc.constant_id);
        if (!specVal.first)
            continue;
        memcpy(&val.m.c[0].r[0].u32, specVal.first, specVal.second); // this will do for spec constants of scalar types (uint, int, float,...) regardless of size
        // but it's ok since ,according to SPIR-V specification, only specialization constants of scalar types are supported (only scalar spec constants can have SpecId decoration)
    }
}

static void reorderBindings(spirv_cross::CompilerGLSL& _comp, const COpenGLPipelineLayout* _layout)
{
    auto sort_ = [&_comp](spirv_cross::SmallVector<spirv_cross::Resource>& res) {
        using R_t = spirv_cross::Resource;
        auto cmp = [&_comp](const R_t& a, const R_t& b) {
            uint32_t a_set = _comp.get_decoration(a.id, spv::DecorationDescriptorSet);
            uint32_t b_set = _comp.get_decoration(b.id, spv::DecorationDescriptorSet);
            if (a_set < b_set)
                return true;
            if (a_set > b_set)
                return false;
            uint32_t a_bnd = _comp.get_decoration(a.id, spv::DecorationBinding);
            uint32_t b_bnd = _comp.get_decoration(b.id, spv::DecorationBinding);
            return a_bnd < b_bnd;
        };
        std::sort(res.begin(), res.end(), cmp);
    };
    auto reorder_ = [&_comp,_layout](spirv_cross::SmallVector<spirv_cross::Resource>& res, asset::E_DESCRIPTOR_TYPE _dtype1, asset::E_DESCRIPTOR_TYPE _dtype2, uint32_t _bndPresum[COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT]) {
        uint32_t availableBinding = 0u;
		uint32_t currSet = 0u;
		const IGPUDescriptorSetLayout* dsl = nullptr;
        for (uint32_t i = 0u, j = 0u; i < res.size(); ++j) {
			uint32_t set = _comp.get_decoration(res[i].id, spv::DecorationDescriptorSet);
			if (!dsl || currSet!=set) {
				currSet = set;
				dsl = _layout->getDescriptorSetLayout(currSet);
				availableBinding = _bndPresum[currSet];
				j = 0u;
			}

            const spirv_cross::Resource& r = res[i];
			const uint32_t r_bnd = _comp.get_decoration(r.id, spv::DecorationBinding);

			const auto& bindInfo = dsl->getBindings().begin()[j];
			if (bindInfo.binding != r_bnd)
			{
				const asset::E_DESCRIPTOR_TYPE dtype = bindInfo.type;
				if (dtype==_dtype1 || dtype==_dtype2)
					availableBinding += bindInfo.count;
				continue;
			}

            _comp.set_decoration(r.id, spv::DecorationBinding, availableBinding);
			availableBinding += bindInfo.count;
            _comp.unset_decoration(r.id, spv::DecorationDescriptorSet);
			++i;
        }
    };

    spirv_cross::ShaderResources resources = _comp.get_shader_resources(_comp.get_active_interface_variables());
	assert(resources.push_constant_buffers.size()<=1u);
	for (const auto& pushConstants : resources.push_constant_buffers)
		_comp.set_decoration(pushConstants.id, spv::DecorationLocation, 0);

	uint32_t presum[COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT]{};
#define UPDATE_PRESUM(_type)\
	for (uint32_t i = 0u; i < COpenGLPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)\
		presum[i] = _layout->getMultibindParamsForDescSet(i)._type.first

	UPDATE_PRESUM(textures);
    auto samplers = std::move(resources.sampled_images);
    for (const auto& samplerBuffer : resources.separate_images)
        if (_comp.get_type(samplerBuffer.type_id).image.dim == spv::DimBuffer)
            samplers.push_back(samplerBuffer);//see https://github.com/KhronosGroup/SPIRV-Cross/wiki/Reflection-API-user-guide for explanation
    sort_(samplers);
    reorder_(samplers, asset::EDT_COMBINED_IMAGE_SAMPLER, asset::EDT_UNIFORM_TEXEL_BUFFER, presum);

	UPDATE_PRESUM(textureImages);
    auto images = std::move(resources.storage_images);
    sort_(images);
    reorder_(images, asset::EDT_STORAGE_IMAGE, asset::EDT_STORAGE_TEXEL_BUFFER, presum);

	UPDATE_PRESUM(ubos);
    auto ubos = std::move(resources.uniform_buffers);
    sort_(ubos);
    reorder_(ubos, asset::EDT_UNIFORM_BUFFER, asset::EDT_UNIFORM_BUFFER_DYNAMIC, presum);

	UPDATE_PRESUM(ssbos);
    auto ssbos = std::move(resources.storage_buffers);
    sort_(ssbos);
    reorder_(ssbos, asset::EDT_STORAGE_BUFFER, asset::EDT_STORAGE_BUFFER_DYNAMIC, presum);
#undef UPDATE_PRESUM
}

static GLenum ESS2GLenum(asset::ISpecializedShader::E_SHADER_STAGE _stage)
{
    using namespace asset;
    switch (_stage)
    {
		case asset::ISpecializedShader::ESS_VERTEX: return GL_VERTEX_SHADER;
		case asset::ISpecializedShader::ESS_TESSELATION_CONTROL: return GL_TESS_CONTROL_SHADER;
		case asset::ISpecializedShader::ESS_TESSELATION_EVALUATION: return GL_TESS_EVALUATION_SHADER;
		case asset::ISpecializedShader::ESS_GEOMETRY: return GL_GEOMETRY_SHADER;
		case asset::ISpecializedShader::ESS_FRAGMENT: return GL_FRAGMENT_SHADER;
		case asset::ISpecializedShader::ESS_COMPUTE: return GL_COMPUTE_SHADER;
		default: return GL_INVALID_ENUM;
    }
}

}//namesapce impl

}
}//nbl::video

using namespace nbl;
using namespace nbl::video;

COpenGLSpecializedShader::COpenGLSpecializedShader(uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializedShader::SInfo& _specInfo, core::vector<SUniform>&& uniformList) :
	core::impl::ResolveAlignment<IGPUSpecializedShader, core::AllocationOverrideBase<128>>(_specInfo.shaderStage),
    m_GLstage(impl::ESS2GLenum(_specInfo.shaderStage)),
	m_specInfo(_specInfo),//TODO make it move()
	m_spirv(core::smart_refctd_ptr<const asset::ICPUBuffer>(_spirv))
{
	m_options.version = _GLSLversion;
	//vulkan_semantics=false causes spirv_cross to translate push_constants into non-UBO uniform of struct type! Exactly like we wanted!
	m_options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of built-in variables renaming, but need testing
	m_options.separate_shader_objects = true;

	core::XXHash_256(_spirv->getPointer(), _spirv->getSize(), m_spirvHash.data());

	m_uniformsList = uniformList;
}

auto COpenGLSpecializedShader::compile(const COpenGLPipelineLayout* _layout, const spirv_cross::ParsedIR* _parsedSpirv) const -> std::pair<GLuint, SProgramBinary>
{
	spirv_cross::ParsedIR parsed;
	if (_parsedSpirv)
		parsed = _parsedSpirv[0];
	else
	{
		spirv_cross::Parser parser(reinterpret_cast<const uint32_t*>(m_spirv->getPointer()), m_spirv->getSize()/4ull);
		parser.parse();
		parsed = std::move(parser.get_parsed_ir());
	}
	spirv_cross::CompilerGLSL comp(std::move(parsed));

	comp.set_entry_point(m_specInfo.entryPoint, asset::ESS2spvExecModel(m_specInfo.shaderStage));
	comp.set_common_options(m_options);

	impl::specialize(comp, m_specInfo);
	impl::reorderBindings(comp, _layout);

	std::string glslCode = comp.compile();
	const char* glslCode_cstr = glslCode.c_str();

	GLuint GLname = COpenGLExtensionHandler::extGlCreateShaderProgramv(m_GLstage, 1u, &glslCode_cstr);

	GLchar logbuf[1u<<12]; //4k
	COpenGLExtensionHandler::extGlGetProgramInfoLog(GLname, sizeof(logbuf), nullptr, logbuf);
	if (logbuf[0])
		os::Printer::log(logbuf, ELL_ERROR);

	if (m_locations.empty())
		gatherUniformLocations(GLname);

	SProgramBinary binary;
	GLint binaryLength = 0;
	COpenGLExtensionHandler::extGlGetProgramiv(GLname, GL_PROGRAM_BINARY_LENGTH, &binaryLength);
	binary.binary = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint8_t>>(binaryLength);
	COpenGLExtensionHandler::extGlGetProgramBinary(GLname, binaryLength, nullptr, &binary.format, binary.binary->data());

	return {GLname, std::move(binary)};
}

void COpenGLSpecializedShader::gatherUniformLocations(GLuint _GLname) const
{
	m_locations.resize(m_uniformsList.size());
	for (size_t i = 0ull; i < m_uniformsList.size(); ++i)
		m_locations[i] = COpenGLExtensionHandler::extGlGetUniformLocation(_GLname, m_uniformsList[i].m.name.c_str());
}

#endif