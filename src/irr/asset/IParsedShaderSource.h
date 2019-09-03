#ifndef __IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__
#define __IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__

#include "irr/core/core.h"
#include "irr/asset/ICPUBuffer.h"

namespace spirv_cross
{
    class ParsedIR;
}

namespace irr
{
namespace asset
{

class IParsedShaderSource : public core::IReferenceCounted
{
	protected:
		virtual ~IParsedShaderSource();

	public:
		struct defer_t {};
		_IRR_STATIC_INLINE_CONSTEXPR defer_t defer = {};

		//! Parsing occurs instantly, during constructor execution
		IParsedShaderSource(const ICPUBuffer* _spirvBytecode);
		//! Parsing occurs upon getUnderlyingRepresentation call
		IParsedShaderSource(const ICPUBuffer* _spirvBytecode, defer_t);

		const spirv_cross::ParsedIR& getUnderlyingRepresentation() const
		{
			parse();
			return *m_parsed; 
		}

	protected:
		void parse() const;

	protected:
		mutable const spirv_cross::ParsedIR* m_parsed;
		//! Raw SPIR-V bytecode
		mutable const ICPUBuffer* m_raw;
};

}}

#endif//__IRR_I_PARSED_SHADER_SOURCE_H_INCLUDED__
