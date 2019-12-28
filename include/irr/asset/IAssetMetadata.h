#ifndef __IRR_I_ASSET_METADATA_H_INCLUDED__
#define __IRR_I_ASSET_METADATA_H_INCLUDED__

#include "irr/core/core.h"

namespace irr
{
namespace asset
{

//! A class managing Asset's metadata context
/**
	Sometimes there may be nedeed attaching some metadata by a Loader
	into Asset structure - that's why the class is defined.

	Pay attention that it hasn't been done exactly yet, engine doesn't provide
	metadata injecting.

	Metadata are extra data retrieved by the loader, which aren't ubiquitously representable by the engine.
	These could be for instance global data about the file or scene, IDs, names, default view/projection,
	complex animations and hierarchies, physics simulation data, AI data, lighting or extra material metadata.

	Flexibility has been provided, it is expected each loader has its own base metadata class implementing the 
	IAssetMetadata interface with its own type enum that other loader's metadata classes derive from the base.
*/
class IAssetMetadata : public core::IReferenceCounted
{
	protected:
		virtual ~IAssetMetadata() = default;

	public:
		//! This could actually be reworked to something more usable
		/*
			To implement by user. Returns a Loader name that may attach some metadata into Asset structure.

			@see IAssetMetadata

			Due to external and custom Asset Loaders static_cast cannot be protected with a type enum comparision, 
			so a string is provided.
		*/
		virtual const char* getLoaderName() const = 0;
};


}
}

#endif // __IRR_I_ASSET_METADATA_H_INCLUDED__
