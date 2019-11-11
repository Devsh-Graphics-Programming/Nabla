#ifndef __IRR_I_SPIR_V_PROGRAM_H_INCLUDED__
#define __IRR_I_SPIR_V_PROGRAM_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"

namespace irr
{
namespace asset
{

class ISPIR_VProgram : public core::IReferenceCounted
{
	protected:
		virtual ~ISPIR_VProgram() = default;

	public:
		ISPIR_VProgram(core::smart_refctd_ptr<ICPUBuffer>&& _bytecode) : m_bytecode{_bytecode}
		{
		}

		const ICPUBuffer* getBytecode() const { return m_bytecode.get(); }

	protected:
		core::smart_refctd_ptr<ICPUBuffer> m_bytecode;
};

}
}

#endif//__IRR_I_SPIR_V_PROGRAM_H_INCLUDED__
