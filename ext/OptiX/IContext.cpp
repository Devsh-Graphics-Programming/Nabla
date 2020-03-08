#include "../../ext/OptiX/IContext.h"

#include "../../ext/OptiX/Manager.h"


using namespace irr;
using namespace asset;
using namespace video;

using namespace irr::ext::OptiX;


const OptixModuleCompileOptions ext::OptiX::IContext::defaultOptixModuleCompileOptions = []() -> OptixModuleCompileOptions
{
	static OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
	return module_compile_options;
}();


core::smart_refctd_ptr<IModule> ext::OptiX::IContext::createModuleFromPTX(	const char* ptx, size_t ptx_size, const OptixPipelineCompileOptions& pipeline_options,
																			const OptixModuleCompileOptions& module_options,
																			std::string* log)
{
	OptixModule module = nullptr;
	size_t sizeof_log = 0u;
	optixModuleCreateFromPTX(
		optixContext,
		&module_options,
		&pipeline_options,
		ptx,
		ptx_size,
		nullptr,
		log ? &sizeof_log:nullptr,
		&module);
	if (log && sizeof_log)
	{
		auto oldLen = log->length();
		log->resize(oldLen+sizeof_log);
		OptixModule tmp = nullptr;
		optixModuleCreateFromPTX(
			optixContext,
			&module_options,
			&pipeline_options,
			ptx,
			ptx_size,
			log->data()+oldLen,
			&sizeof_log,
			&tmp);
		if (tmp)
			optixModuleDestroy(tmp);
	}
	return core::smart_refctd_ptr<IModule>(new IModule(module),core::dont_grab);
}

core::smart_refctd_ptr<IModule> ext::OptiX::IContext::compileModuleFromSource_helper(	const char* source, const char* filename,
																						const OptixPipelineCompileOptions& pipeline_options,
																						const OptixModuleCompileOptions& module_options,
																						const char* const* compile_options_begin, const char* const* compile_options_end,
																						std::string* log)
{
	auto ptx = manager->compileOptiXProgram<core::SRange<const char* const> >(source,filename,{compile_options_begin,compile_options_end},log);
	return createModuleFromPTX(ptx,pipeline_options,module_options,log);
}