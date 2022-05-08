#ifndef __I_CACHE_KEY_CREATOR_H_INCLUDED__
#define __I_CACHE_KEY_CREATOR_H_INCLUDED__

#include "nbl/asset/IAsset.h"

namespace nbl
{
	namespace asset
	{
		class NBL_API ICacheKeyCreator
		{
			public:
				virtual std::string to_string() = 0;

				template<typename Type>
				_NBL_STATIC_INLINE std::string getHexString(const Type& value)
				{
					std::stringstream stream;
					stream << std::setfill('0') << std::hex << value;
					return stream.str();
				};
		};
	}
}

#endif // __I_CACHE_KEY_CREATOR_H_INCLUDED__
