// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CommandLineHandler.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

using namespace nbl;
using namespace asset;
using namespace core;

CommandLineHandler::CommandLineHandler(const std::vector<std::string>& argv)
{
	if(argv.size() > MaxRayTracerCommandLineArgs)
	{
		std::cout << helpMessage.data() << std::endl;
		return;
	}
	
	auto logError = [&](const std::string message)
	{
		std::cout << "ERROR (" + std::to_string(__LINE__) + " line): " + message << std::endl;
	};

	auto arguments = argv;

	auto getSerializedValues = [&](const auto& variablesStream, const std::regex& separator=std::regex{"[[:s:]]"})
	{
		std::sregex_token_iterator it{ variablesStream.begin(), variablesStream.end(), separator, -1 };
		std::vector<std::string> variablesHandle = { it,{} };

		// remove any accidental whitespace only vars
		variablesHandle.erase(
			std::remove_if(
				variablesHandle.begin(),variablesHandle.end(),
				[](const std::string& x) {return !std::regex_search(x,std::regex{"[^[:s:]]"}); }
			),
			variablesHandle.end()
		);

		// remove double-quotes
		for(auto& var : variablesHandle)
		{
			var.erase(std::remove(var.begin(), var.end(), '\"'), var.end());
		}

		return variablesHandle;
	};

	initializeMatchingMap();

	RaytracerExampleArguments previousArg = REA_COUNT;

	bool success = true;

	for (auto i = 0; i < arguments.size(); ++i)
	{
		std::string rawFetchedCmdArgument = arguments[i];
		
		bool addToPreviousOption = false;

		const auto firstHyphen = rawFetchedCmdArgument.find_first_of("-");
		if(firstHyphen != 0)
			addToPreviousOption = true;

		if(addToPreviousOption)
		{
			if(REA_COUNT != previousArg)
			{
				if(!rawVariables[previousArg].has_value())
					rawVariables[previousArg].emplace(std::vector<std::string>());
				
				auto & outVector = rawVariables[previousArg].value();
				std::vector<std::string> toAdd = getSerializedValues(rawFetchedCmdArgument);
				outVector.insert(outVector.end(), toAdd.begin(), toAdd.end());
			}
			else
			{
				logError("Unexcepted argument!, command options should start with '-' character");
				success = false;
				break;
			}
		}
		else
		{
			const auto offset = firstHyphen + 1;
			const auto endOfFetchedVariableName = rawFetchedCmdArgument.find_first_of("=");
			const auto count = endOfFetchedVariableName - offset;
			const auto cmdFetchedVariable = rawFetchedCmdArgument.substr(offset, count);
			std::string variable = cmdFetchedVariable;
			auto arg = getMatchedVariableMapID(variable);
			
			if(arg == REA_COUNT)
			{
				logError("Unexcepted argument!!");
				success = false;
				break;
			}

			if(rawVariables[arg].has_value())
			{
				logError("Variable used previously!");
				success = false;
				break;
			}
			
			if(endOfFetchedVariableName != std::string::npos)
			{
				auto value = rawFetchedCmdArgument.substr(endOfFetchedVariableName + 1);
				std::vector<std::string> toAdd = getSerializedValues(value);
				rawVariables[arg].emplace(toAdd);
			}
			else
			{
				std::vector<std::string> emptyVec;
				rawVariables[arg].emplace(emptyVec);
			}

			previousArg = arg;
		}

	}

	if (!validateParameters() || !success)
		return;

	performFinalAssignmentStepForUsefulVariables();
}

bool CommandLineHandler::validateParameters()
{
	auto logError = [&](const std::string message)
	{
		std::cout << "ERROR (" + std::to_string(__LINE__) + " line): " + message << std::endl;
	};

	if(rawVariables[REA_SCENE].has_value())
	{
		auto sceneDirectory = rawVariables[REA_SCENE].value();
		if(sceneDirectory.empty())
		{
			logError("Expected at least one value for SCENE");
			return false;
		}
	}

	return true;
}