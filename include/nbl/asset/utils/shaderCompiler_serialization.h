#include "nbl/asset/utils/IShaderCompiler.h"


// -------------------------------------------- JSON loading and storing logic for all necessary structs -----------------------------------------------------------
namespace JS
{
	using SCompilerData = nbl::asset::IShaderCompiler::CCache::SEntry::SCompilerData;

	template<>
	struct TypeHandler<nbl::asset::IShader::E_SHADER_STAGE>
	{
		static inline Error to(nbl::asset::IShader::E_SHADER_STAGE& to_type, ParseContext& context)
		{
			uint32_t aux;
			auto retVal = TypeHandler<uint32_t>::to(aux, context);
			to_type = static_cast<nbl::asset::IShader::E_SHADER_STAGE>(aux);
			return retVal;
		}

		static inline void from(const nbl::asset::IShader::E_SHADER_STAGE& from_type, Token& token, Serializer& serializer)
		{
			return TypeHandler<uint32_t>::from(static_cast<uint32_t>(from_type), token, serializer);
		}
	};

	template<>
	struct TypeHandler<nbl::asset::IShaderCompiler::E_SPIRV_VERSION>
	{
		static inline Error to(nbl::asset::IShaderCompiler::E_SPIRV_VERSION& to_type, ParseContext& context)
		{
			uint32_t aux;
			auto retVal = TypeHandler<uint32_t>::to(aux, context);
			to_type = static_cast<nbl::asset::IShaderCompiler::E_SPIRV_VERSION>(aux);
			return retVal;
		}

		static inline void from(const nbl::asset::IShaderCompiler::E_SPIRV_VERSION& from_type, Token& token, Serializer& serializer)
		{
			return TypeHandler<uint32_t>::from(static_cast<uint32_t>(from_type), token, serializer);
		}
	};

	template<>
	struct TypeHandler<nbl::core::bitflag<nbl::asset::IShaderCompiler::E_DEBUG_INFO_FLAGS>>
	{
		static inline Error to(nbl::core::bitflag<nbl::asset::IShaderCompiler::E_DEBUG_INFO_FLAGS>& to_type, ParseContext& context)
		{
			uint8_t aux;
			auto retVal = TypeHandler<uint8_t>::to(aux, context);
			to_type = nbl::core::bitflag<nbl::asset::IShaderCompiler::E_DEBUG_INFO_FLAGS>(aux);
			return retVal;
		}

		static inline void from(const nbl::core::bitflag<nbl::asset::IShaderCompiler::E_DEBUG_INFO_FLAGS>& from_type, Token& token, Serializer& serializer)
		{
			return TypeHandler<uint8_t>::from(static_cast<uint8_t>(from_type.value), token, serializer);
		}
	};

	template<>
	struct TypeHandler<nbl::asset::ISPIRVOptimizer::E_OPTIMIZER_PASS>
	{
		static inline Error to(nbl::asset::ISPIRVOptimizer::E_OPTIMIZER_PASS& to_type, ParseContext& context)
		{
			uint8_t aux;
			auto retVal = TypeHandler<uint8_t>::to(aux, context);
			to_type = static_cast<nbl::asset::ISPIRVOptimizer::E_OPTIMIZER_PASS>(aux);
			return retVal;
		}

		static inline void from(const nbl::asset::ISPIRVOptimizer::E_OPTIMIZER_PASS& from_type, Token& token, Serializer& serializer)
		{
			return TypeHandler<uint8_t>::from(static_cast<uint8_t>(from_type), token, serializer);
		}
	};

	template<>
	struct TypeHandler<nbl::system::path>
	{
		static inline Error to(nbl::system::path& to_type, ParseContext& context)
		{
			std::string aux;
			auto retVal = TypeHandler<std::string>::to(aux, context);
			to_type = std::move(aux);
			return retVal;
		}

		static inline void from(const nbl::system::path& from_type, Token& token, Serializer& serializer)
		{
			std::string aux = from_type.string();
			return TypeHandler<std::string>::from(aux, token, serializer);
		}
	};

	template<unsigned int N>
	struct TypeHandler<std::array<uint64_t, N>>
	{
		static inline Error to(std::array<uint64_t, N>& to_type, ParseContext& context)
		{
			std::vector<uint64_t> aux(N);
			auto retVal = TypeHandler<std::vector<uint64_t>>::to(aux, context);
			std::move(aux.begin(), aux.end(), to_type.begin());
			return retVal;
		}

		static inline void from(const std::array<uint64_t, N>& from_type, Token& token, Serializer& serializer)
		{
			std::vector<uint64_t> aux(N);
			std::copy_n(from_type.begin(), N, aux.begin());
			return TypeHandler<std::vector<uint64_t>>::from(aux, token, serializer);
		}
	};
}

JS_OBJECT_EXTERNAL(nbl::asset::IShaderCompiler::CCache::SEntry::SCompilerData, JS_MEMBER(stage), JS_MEMBER(targetSpirvVersion), JS_MEMBER(optimizerPasses), JS_MEMBER(debugInfoFlags), JS_MEMBER(preprocessorData));

JS_OBJECT_EXTERNAL(nbl::asset::IShaderCompiler::CCache::SEntry::SDependency, JS_MEMBER(requestingSourceDir), JS_MEMBER(identifier), JS_MEMBER(contents), JS_MEMBER(hash), JS_MEMBER(standardInclude));

namespace JS {
	using SDependency = nbl::asset::IShaderCompiler::CCache::SEntry::SDependency;
	using dependency_container_t = core::smart_refctd_dynamic_array<const SDependency>;
	template<>
	struct TypeHandler<dependency_container_t>
	{
		static inline Error to(dependency_container_t& to_type, ParseContext& context)
		{
			std::vector<SDependency> aux;
			auto retVal = TypeHandler<std::vector<SDependency>>::to(aux, context);
			to_type = core::make_refctd_dynamic_array<dependency_container_t>(aux);
			return retVal;
		}

		static inline void from(const dependency_container_t& from_type, Token& token, Serializer& serializer)
		{
			std::vector<SDependency> aux;
			for (SDependency& dep : from_type) {
				aux.push_back(dep);
			}
			return TypeHandler<std::vector<SDependency>>::from(aux, token, serializer);
		}
	};
}

JS_OBJECT_EXTERNAL(nbl::asset::IShaderCompiler::CCache::SEntry, JS_MEMBER(mainFilePath), JS_MEMBER(mainFileHash), JS_MEMBER(compilerData), JS_MEMBER(compilerDataHash), JS_MEMBER(shaderStorePath), JS_MEMBER(dependenciesPath));