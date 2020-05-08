#include "../ext/AutoExposure/CToneMapper.h"


using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::AutoExposure;


constexpr uint32_t DISPATCH_SIZE = 16u;


core::smart_refctd_ptr<CToneMapper> CToneMapper::create(IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler)
{
	if (!_driver)
		return nullptr;
	if (inputFormat!=asset::EF_R16G16B16A16_SFLOAT)
		return nullptr;

	IGPUDescriptorSetLayout::SBinding bnd[3];
	bnd[0].binding = 0u;
	bnd[0].type = asset::EDT_UNIFORM_BUFFER_DYNAMIC;
	bnd[0].count = 1u;
	bnd[0].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
	bnd[0].samplers = nullptr;
	bnd[1].binding = 1u;
	bnd[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
	bnd[1].count = 1u;
	bnd[1].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
	bnd[1].samplers = nullptr;
	bnd[2].binding = 2u;
	bnd[2].type = asset::EDT_STORAGE_IMAGE;
	bnd[2].count = 1u;
	bnd[2].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
	bnd[2].samplers = nullptr;
	auto dsLayout = _driver->createGPUDescriptorSetLayout(bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding));

	auto pipelineLayout = _driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(dsLayout));

	std::ostringstream glsl;
	glsl << "#version 430 core\n";
	glsl << "layout (local_size_x = "<<DISPATCH_SIZE<<", local_size_y = "<<DISPATCH_SIZE<<") in;\n";
	glsl << "layout(set=3, binding=1) uniform usampler2D inImage;\n";
	glsl << "layout(set=3, binding=2, r32ui) uniform writeonly uimage2D outImage;\n";
	glsl << getInclude();
	glsl << R"===(layout(set=3, binding=0) uniform TonemappingParameters
{
	irr_ext_Autoexposure_ReinhardParams params;
};

void main()
{
	//uint data = textureGatherOffset(); // later optimization

	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(uv,textureSize(inImage,0))))
		return;

	uvec2 data = texelFetch(inImage,uv,0).rg;
	vec3 hdr = vec3(unpackHalf2x16(data[0]).rg,unpackHalf2x16(data[1])[0]);
	vec4 ldr = vec4(irr_ext_Autoexposure_ToneMapReinhard(params,hdr),1.0);
	// TODO: Add dithering
	imageStore(outImage,uv,uvec4(packUnorm4x8(ldr),0u,0u,0u));
}
	)===";
	auto spirv = compiler->createSPIRVFromGLSL(glsl.str().c_str(),asset::ISpecializedShader::ESS_COMPUTE,"main","CToneMapper");
	auto shader = _driver->createGPUShader(std::move(spirv));
	
	asset::ISpecializedShader::SInfo specInfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE);

	auto computePipeline = _driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), _driver->createGPUSpecializedShader(shader.get(),std::move(specInfo)));

	auto tmp = new CToneMapper(_driver,inputFormat,std::move(dsLayout),std::move(pipelineLayout),std::move(computePipeline));
	return core::smart_refctd_ptr<CToneMapper>(tmp,core::dont_grab);
}

CToneMapper::CToneMapper(	IVideoDriver* _driver, asset::E_FORMAT inputFormat,
							core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& _dsLayout,
							core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _pipelineLayout,
							core::smart_refctd_ptr<video::IGPUComputePipeline>&& _computePipeline) :
									m_driver(_driver), format(inputFormat), dsLayout(std::move(_dsLayout)),
									pipelineLayout(std::move(_pipelineLayout)), computePipeline(std::move(_computePipeline))
{
	if (format==asset::EF_R16G16B16A16_SFLOAT)
		viewFormat = asset::EF_R32G32_UINT;
}

bool CToneMapper::tonemap(video::IGPUImageView* inputThatsInTheSet, video::IGPUDescriptorSet* set, uint32_t parameterUBOOffset)
{
	const auto& params = inputThatsInTheSet->getCreationParameters();
	if (params.format!=viewFormat)
		return false;
	if (params.image->getCreationParameters().format!=format)
		return false;

	auto offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(1u);
	offsets->operator[](0u) = parameterUBOOffset;

	m_driver->bindComputePipeline(computePipeline.get());
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE,pipelineLayout.get(),3,1,const_cast<const video::IGPUDescriptorSet**>(&set),&offsets);

	auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
	imgViewSize /= DISPATCH_SIZE;
	m_driver->dispatch(imgViewSize.x,imgViewSize.y,1u);
	return true;
}

#if 0
//#define PROFILE_TONEMAPPER

bool CToneMapper::CalculateFrameExposureFactors(video::IGPUBuffer* outBuffer, video::IGPUBuffer* uniformBuffer, core::smart_refctd_ptr<video::ITexture>&& inputTexture)
{
    bool highRes = false;
    if (!inputTexture)
        return false;

    video::COpenGLTexture* asGlTex = dynamic_cast<video::COpenGLTexture*>(inputTexture.get());
    if (asGlTex->getOpenGLTextureType()!=GL_TEXTURE_2D)
        return false;

    GLint prevProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProgram);


#ifdef PROFILE_TONEMAPPER
    video::IQueryObject* timeQuery = m_driver->createElapsedTimeQuery();
    m_driver->beginQuery(timeQuery);
#endif // PROFILE_TONEMAPPER

    video::STextureSamplingParams params;
    params.MaxFilter = video::ETFT_LINEAR_NO_MIP;
    params.MinFilter = video::ETFT_LINEAR_NO_MIP;
    params.UseMipmaps = 0;

    const video::COpenGLDriver::SAuxContext* foundConst = static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext();
    video::COpenGLDriver::SAuxContext* found = const_cast<video::COpenGLDriver::SAuxContext*>(foundConst);
    found->setActiveTexture(0,std::move(inputTexture),params);


    video::COpenGLExtensionHandler::extGlUseProgram(m_histogramProgram);

    const video::COpenGLBuffer* buffers[2] = {static_cast<const video::COpenGLBuffer*>(m_histogramBuffer),static_cast<const video::COpenGLBuffer*>(outBuffer)};
    ptrdiff_t offsets[2] = {0,0};
    ptrdiff_t sizes[2] = {m_histogramBuffer->getSize(),outBuffer->getSize()};
    found->setActiveSSBO(0,2,buffers,offsets,sizes);

    buffers[0] = static_cast<const video::COpenGLBuffer*>(uniformBuffer);
    sizes[0] = uniformBuffer->getSize();
    found->setActiveUBO(0,1,buffers,offsets,sizes);

    video::COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0],m_workGroupCount[1],1);
    video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    video::COpenGLExtensionHandler::extGlUseProgram(m_autoExpParamProgram);
    video::COpenGLExtensionHandler::pGlDispatchCompute(1,1, 1);


    video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
    video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

#ifdef PROFILE_TONEMAPPER
    m_driver->endQuery(timeQuery);
    uint32_t timeTaken=0;
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::AutoExposure CS Time Taken:", std::to_string(timeTaken).c_str(),ELL_ERROR);
#endif // PROFILE_TONEMAPPER

    return true;
}
#endif