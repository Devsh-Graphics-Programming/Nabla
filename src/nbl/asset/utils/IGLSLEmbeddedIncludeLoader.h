// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__
#define __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__

#include "nbl/system/declarations.h"

#include "nbl/asset/utils/IBuiltinIncludeLoader.h"

namespace nbl::asset
{

class IGLSLEmbeddedIncludeLoader : public IBuiltinIncludeLoader
{
	protected:
		virtual ~IGLSLEmbeddedIncludeLoader() = default;

		inline core::vector<std::pair<std::regex,HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			std::string pattern(getVirtualDirectoryName());
			pattern += ".*";
			HandleFunc_t tmp = [this](const std::string& _name) -> std::string {
				return getFromDiskOrEmbedding(_name);
			};
			return {{std::regex{pattern},std::move(tmp)}};
		}
		
		static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
		{
			core::vector<std::string> args;

			std::stringstream ss{ _path };
			std::string arg;
			while (std::getline(ss, arg, '/'))
				args.push_back(std::move(arg));

			return args;
		}
		
		system::ISystem* s;

	public:
		IGLSLEmbeddedIncludeLoader(system::ISystem* system) : s(system) {}

		//
		const char* getVirtualDirectoryName() const override { return ""; }

		//
		inline std::string getFromDiskOrEmbedding(const std::string& _name) const
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			auto path = "nbl/builtin/" + _name;
			s->createFile(future,path,core::bitflag(system::IFileBase::ECF_READ)|system::IFileBase::ECF_MAPPABLE);
			core::smart_refctd_ptr<const system::IFile> data = future.get();
			if (!data)
				return "";
			auto begin = reinterpret_cast<const char*>(data->getMappedPointer());
			auto end = begin + data->getSize();
			return std::string(begin, end);
		}
};

}

#endif
