// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_APPLICATION_TEMPLATES_MONO_ASSET_MANAGER_APPLICATION_HPP_INCLUDED_
#define _NBL_APPLICATION_TEMPLATES_MONO_ASSET_MANAGER_APPLICATION_HPP_INCLUDED_


// we need a system and a logger
#include "nbl/application_templates/MonoSystemMonoLoggerApplication.hpp"


namespace nbl::application_templates
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoAssetManagerApplication : public virtual MonoSystemMonoLoggerApplication
{
		using base_t = MonoSystemMonoLoggerApplication;

	public:
		using base_t::base_t;

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		MonoAssetManagerApplication() = default;

		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			using namespace core;
			m_assetMgr = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));

			return true;
		}

		core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;
};

}
#endif