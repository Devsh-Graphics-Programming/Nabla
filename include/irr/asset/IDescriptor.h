#ifndef __IRR_I_DESCRIPTOR_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace asset
{

class IDescriptor : public virtual core::IReferenceCounted
{
	public:
		enum E_CATEGORY
		{
			EC_BUFFER,
			EC_IMAGE,
			EC_BUFFER_VIEW
		};

		virtual E_CATEGORY getTypeCategory() const = 0;

	protected:
		virtual ~IDescriptor() = default;
};

}
}

#endif