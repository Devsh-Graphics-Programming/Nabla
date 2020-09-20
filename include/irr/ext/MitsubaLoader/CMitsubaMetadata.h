#ifndef __IRR_C_MITSUBA_METADATA_H_INCLUDED__
#define __IRR_C_MITSUBA_METADATA_H_INCLUDED__

#include "irr/asset/IAssetMetadata.h"
#include "irr/ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

namespace irr
{
	namespace ext
	{
		namespace MitsubaLoader
		{
			//! A class to derive mitsuba mesh loader metadata objects from

			class CMitsubaMetadata : public asset::IAssetMetadata
			{
				public:

					CMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata> _mitsubaMetadata) : mitsubaMetadata(std::move(_mitsubaMetadata)) {}

					std::string getName() const
					{
						return name;
					}

					auto getMitsubaMetadata() const { return mitsubaMetadata; }

					_IRR_STATIC_INLINE_CONSTEXPR const char* loaderName = "CMitsubaLoader";
					const char* getLoaderName() const override { return loaderName; }

				private:

					std::string name;
					core::smart_refctd_ptr<CGlobalMitsubaMetadata> mitsubaMetadata;
			};
		}
	}
}

#endif // __IRR_C_MITSUBA_METADATA_H_INCLUDED__
