// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CommandLineHandler.hpp"

#include <algorithm>

using namespace nbl;
using namespace asset;
using namespace core;

CommandLineHandler::CommandLineHandler(core::vector<std::string> argv, IAssetManager* am, nbl::io::IFileSystem* fs) : status(false), assetManager(am)
{
	auto startEntireTime = std::chrono::steady_clock::now();

	if(argv.size()>=MANDATORY_CMD_ARGUMENTS_AMOUNT && argv.size()<=PROPER_CMD_ARGUMENTS_AMOUNT)
	{
		os::Printer::log("Confirm input from Commandline arguments", ELL_INFORMATION);
		mode = CLM_CMD_LIST;
	}
	else if (argv.size()>=PROPER_BATCH_FILE_ARGUMENTS_AMOUNT)
	{
		os::Printer::log("Confirm input from Batch File arguments", ELL_INFORMATION);
		mode = CLM_BATCH_INPUT;
	}
	else
	{
		mode = CLM_UNKNOWN;
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		return;
	}
	
	auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(am,fs);
	mitsubaLoader->initialize();
	am->addAssetLoader(std::move(mitsubaLoader));

	core::vector<std::array<std::string, PROPER_CMD_ARGUMENTS_AMOUNT>> argvMappedList;

	auto pushArgvList = [&](auto argvStream, auto variableCount)
	{
		auto& back = argvMappedList.emplace_back();
		for (auto i = 0; i < variableCount; ++i)
			back[i] = argvStream[i];
	};

	auto getBatchFilesArgvStream = [&](std::string& fileStream) -> std::vector<std::string>
	{
		std::regex regex{"^"};
		std::sregex_token_iterator it{ fileStream.begin(), fileStream.end(), regex, -1 };
		return { it, {} };
	};

	auto getSerializedValues = [&](const auto& variablesStream, auto supposedArgumentsAmout, const std::regex& separator=std::regex{"[[:s:]]"})
	{
		std::sregex_token_iterator it{ variablesStream.begin(), variablesStream.end(), separator, -1 };
		core::vector<std::string> variablesHandle = { it,{} };

		// remove any accidental whitespace only vars
		variablesHandle.erase(
			std::remove_if(
				variablesHandle.begin(),variablesHandle.end(),
				[](const std::string& x) {return !std::regex_search(x,std::regex{"[^[:s:]]"}); }
			),
			variablesHandle.end()
		);

		return variablesHandle;
	};

	switch (mode)
	{
		case CLM_BATCH_INPUT:
		{
			auto file = am->getFileSystem()->createAndOpenFile(argv[2].c_str());
			std::string fileStream;
			fileStream.resize(file->getSize(), ' ');
			file->read(fileStream.data(), file->getSize());
			fileStream += "\r\n";

			bool error = false;
			const auto batchInputStream = getBatchFilesArgvStream(fileStream);

			for (auto i = 0ul; i < batchInputStream.size(); ++i)
			{
				const auto argvStream = *(batchInputStream.begin() + i);
				// protection against empty lines
				if (!std::regex_search(argvStream, std::regex{ "[^[:s:]]" }))
					continue;

				const auto arguments = getSerializedValues(argvStream, PROPER_CMD_ARGUMENTS_AMOUNT);

				if (arguments.size() < MANDATORY_CMD_ARGUMENTS_AMOUNT || arguments.size() > PROPER_CMD_ARGUMENTS_AMOUNT)
				{
					error = true;
					break;
				}

				pushArgvList(arguments, arguments.size());
			}

			if (error)
			{
				os::Printer::log(requiredArgumentsMessage.data(), ELL_ERROR);
				return;
			}

			break;
		}
		case CLM_CMD_LIST:
		{
			pushArgvList(argv, argv.size());
			break;
		}
		default:
		{
			os::Printer::log("Invalid syntax!", ELL_ERROR);
			os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
			break;
		}
	}

	rawVariables.resize(argvMappedList.size());
	for (auto inputBatchStride = 0ul; inputBatchStride < argvMappedList.size(); ++inputBatchStride)
	{
		const auto cmdArgumentsPerFile = *(argvMappedList.begin() + inputBatchStride);
		auto& rawVariablesHandle = rawVariables[inputBatchStride]; // unorederd map of variables
		initializeMatchingMap(rawVariablesHandle);

		for (auto argumentIterator = 0; argumentIterator < cmdArgumentsPerFile.size(); ++argumentIterator)
		{
			std::string rawFetchedCmdArgument = cmdArgumentsPerFile[argumentIterator];

			const auto offset = rawFetchedCmdArgument.find_first_of("-") + 1;
			const auto endOfFetchedVariableName = rawFetchedCmdArgument.find_first_of("=");
			const auto count = endOfFetchedVariableName - offset;
			const auto cmdFetchedVariable = rawFetchedCmdArgument.substr(offset, count);
			{
				std::string variable = cmdFetchedVariable;
				auto matchedVariableID = getMatchedVariableMapID(variable);

				if (matchedVariableID == DTEA_COUNT)
					continue;

				const auto beginningOfVariables = rawFetchedCmdArgument.find_last_of("=") + 1;
				auto variablesStream = rawFetchedCmdArgument.substr(beginningOfVariables);

				auto forceOutsideAssignment = [&](DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS argument, nbl::core::vector<std::string>& variablesHandle)
				{
					auto& reference = rawVariablesHandle[argument];
					return reference.emplace(variablesHandle);
				};

				auto assignToMap = [&](DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS argument, size_t reservedSpace = 1)
				{
					auto variablesHandle = getSerializedValues(variablesStream, reservedSpace, std::regex{"\,"});
					return forceOutsideAssignment(argument, variablesHandle);
				};

				if (variable == TONEMAPPER)
				{
					auto begin = rawFetchedCmdArgument.find_first_of('=')+1u;
					auto end = rawFetchedCmdArgument.find_first_of('=',begin);
					auto foundOperator = rawFetchedCmdArgument.substr(begin,end-begin);
					static const core::set<std::string> acceptedOperators = { std::string(REINHARD),std::string(ACES),std::string(NONE) };

					if (acceptedOperators.find(foundOperator)!=acceptedOperators.end())
						variable = foundOperator;
					else
					{
						os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): Invalid tonemapper specified! Id of input stride: " + std::to_string(inputBatchStride), ELL_ERROR);
						return;
					}
				}

				bool status = true;
				if (variable == COLOR_FILE)
					assignToMap(DTEA_COLOR_FILE);
				else if (variable == CAMERA_TRANSFORM)
					assignToMap(DTEA_CAMERA_TRANSFORM);
				else if (variable == DENOISER_EXPOSURE_BIAS)
					assignToMap(DTEA_DENOISER_EXPOSURE_BIAS);
				else if (variable == DENOISER_BLEND_FACTOR)
					assignToMap(DTEA_DENOISER_BLEND_FACTOR);
				else if (variable == BLOOM_PSF_FILE)
					assignToMap(DTEA_BLOOM_PSF_FILE);
				else if (variable == BLOOM_RELATIVE_SCALE)
					assignToMap(DTEA_BLOOM_RELATIVE_SCALE);
				else if (variable == BLOOM_INTENSITY)
					assignToMap(DTEA_BLOOM_INTENSITY);
				else if (variable == REINHARD)
					assignToMap(DTEA_TONEMAPPER_REINHARD, 2);
				else if (variable == ACES)
					assignToMap(DTEA_TONEMAPPER_ACES, 2);
				else if (variable == NONE)
					assignToMap(DTEA_TONEMAPPER_NONE, 1);
				else if (variable == OUTPUT)
					assignToMap(DTEA_OUTPUT);
				else if (variable == ALBEDO_FILE)
					assignToMap(DTEA_ALBEDO_FILE);
				else if (variable == NORMAL_FILE)
					assignToMap(DTEA_NORMAL_FILE);
				else if (variable == COLOR_CHANNEL_NAME)
					assignToMap(DTEA_COLOR_CHANNEL_NAME);
				else if (variable == ALBEDO_CHANNEL_NAME)
					assignToMap(DTEA_ALBEDO_CHANNEL_NAME);
				else if (variable == NORMAL_CHANNEL_NAME)
					assignToMap(DTEA_NORMAL_CHANNEL_NAME);
				else
				{
					os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): Unexcepted argument! Id of input stride: " + std::to_string(inputBatchStride), ELL_ERROR);
					assert(status = false);
				}
			}
		}

		if (!validateMandatoryParameters(rawVariablesHandle, inputBatchStride))
			return;
	}

	performFInalAssignmentStepForUsefulVariables();

	auto endEntireTime = std::chrono::steady_clock::now();
	elapsedTimeEntireLoading = endEntireTime - startEntireTime;
	status = true;

	#ifdef _NBL_DEBUG
	os::Printer::log("INFO (" + std::to_string(__LINE__) + " line): Elapsed time of loading entire data: " + std::to_string(elapsedTimeEntireLoading.count()) + " nanoseconds", ELL_INFORMATION);
	os::Printer::log("INFO (" + std::to_string(__LINE__) + " line): Elapsed time of loading without mitsuba xml files: " + std::to_string(elapsedTimeEntireLoading.count() - elapsedTimeXmls.count()) + " nanoseconds", ELL_INFORMATION);
	os::Printer::log("INFO (" + std::to_string(__LINE__) + " line): Elapsed time of loading mitsuba xml files: " + std::to_string(elapsedTimeXmls.count()) + " nanoseconds", ELL_INFORMATION);
	#endif // _NBL_DEBUG
}

bool CommandLineHandler::validateMandatoryParameters(const variablesType& rawVariablesPerFile, const size_t idOfInput)
{
	static const nbl::core::vector<DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS> mandatoryArgumentsOrdinary = { DTEA_COLOR_FILE, DTEA_CAMERA_TRANSFORM, DTEA_DENOISER_EXPOSURE_BIAS, DTEA_DENOISER_BLEND_FACTOR, DTEA_BLOOM_PSF_FILE, DTEA_BLOOM_RELATIVE_SCALE, DTEA_BLOOM_INTENSITY, DTEA_OUTPUT };

	auto log = [&](bool status, const std::string message)
	{
		os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): " + message + " Id of input stride: " + std::to_string(idOfInput), ELL_ERROR);
		//assert(status);
	};

	auto validateOrdinary = [&](const DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS argument)
	{
		if (!rawVariablesPerFile.at(argument).has_value())
			return false;

		//rawVariablesPerFile.at(argument)
		//switch(argument)

		return true;
	};

	auto validateTonemapper = [&]()
	{
		uint32_t tonemapperCount = 0u;
		uint32_t j = DTEA_TONEMAPPER_REINHARD;
		for (DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS num; j<=DTEA_TONEMAPPER_NONE; j++)
		{
			num = (DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS)j;
			auto& opt = rawVariablesPerFile.at(num);
			if (!opt.has_value())
				continue;
			tonemapperCount++;
			if (num!=DTEA_TONEMAPPER_NONE)
			{
				if (opt.value().size() < 2)
				{
					log(status = false, "A non-None tonemapper was not provided with at least 2 arguments!");
					return false;
				}
			}
			else if (opt.value().size() < 1)
			{
				log(status = false, "The None tonemapper was not provided with at least 1 argument!");
				return false;
			}
		}

		if (tonemapperCount==0u)
		{
			log(status = false, "Only one tonemapper must be specified!");
			return false;
		}
		else if (tonemapperCount>1)
		{
			log(status = false, "Only one tonemapper must be specified!");
			return false;
		}

		return true;
	};

	for (const auto& mandatory : mandatoryArgumentsOrdinary)
	{
		bool status = validateOrdinary(mandatory);
		if (!status)
		{
			log(status, "Mandatory argument missing or it doesn't contain any value!");
			return false;
		}
	}

	return validateTonemapper();
}

std::optional<std::string> CommandLineHandler::getNormalFileName(uint64_t id)
{
	bool ableToReturn = rawVariables[id][DTEA_NORMAL_FILE].has_value() && !rawVariables[id][DTEA_NORMAL_FILE].value().empty();
	if (ableToReturn)
	{
		ableToReturn = rawVariables[id][DTEA_ALBEDO_FILE].has_value() && !rawVariables[id][DTEA_ALBEDO_FILE].value().empty();
		if (!ableToReturn)
		{
			os::Printer::log("WARNING (" + std::to_string(__LINE__) + " line): Couldn't accept normal file due to lack of albedo file! Id of input stride: " + std::to_string(id), ELL_WARNING);
			return {};
		}
	}
	else
		return {};

	return rawVariables[id][DTEA_NORMAL_FILE].value()[0];
}

nbl::core::matrix3x4SIMD CommandLineHandler::getCameraTransform(uint64_t id)
{
	static const IAssetLoader::SAssetLoadParams mitsubaLoaderParams = { 0, nullptr, IAssetLoader::ECF_CACHE_EVERYTHING, nullptr, IAssetLoader::ELPF_LOAD_METADATA_ONLY };

	auto getMatrixFromFile = [&]()
	{
		const std::string& filePath = rawVariables[id][DTEA_CAMERA_TRANSFORM].value()[0];

		auto startTime = std::chrono::steady_clock::now();
		auto meshes_bundle = assetManager->getAsset(filePath.data(), mitsubaLoaderParams);
		if (meshes_bundle.getContents().empty())
		{
			os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): The xml file is invalid/cannot be loaded! Id of input file: " + std::to_string(id) + ". File path: " + filePath, ELL_ERROR);
			exit(-1);
		}
		auto endTime = std::chrono::steady_clock::now();
		elapsedTimeXmls += (endTime - startTime);
		
		auto mesh = meshes_bundle.getContents().begin()[0];
		auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());
		const auto mitsubaMetadata = static_cast<const ext::MitsubaLoader::CMitsubaMetadata*>(meshes_bundle.getMetadata());

		bool validateFlag = mitsubaMetadata->m_global.m_sensors.empty();
		if (validateFlag)
		{
			os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): The is no transform matrix in " + filePath + " ! Id of input stride: " + std::to_string(id), ELL_ERROR);
			exit(-2);
		}

		auto transformReference = mitsubaMetadata->m_global.m_sensors[0].transform.matrix.extractSub3x4();
		transformReference.setTranslation(core::vectorSIMDf(0, 0, 0, 0));

		return transformReference;
	};

	auto getMatrixFromSerializedValues = [&]()
	{
		nbl::core::matrix3x4SIMD cameraTransform;
		const auto send = rawVariables[id][DTEA_CAMERA_TRANSFORM].value().end();
		auto sit = rawVariables[id][DTEA_CAMERA_TRANSFORM].value().begin();
		for (auto i = 0; i < 3u && sit != send; i++)
			for (auto j = 0; j < 3u && sit != send; j++)
				cameraTransform(i, j) = std::stof(*(sit++));

		cameraTransform.rows[1] *= vectorSIMDf(-1.f, -1.f, -1.f, 1.f);

		return cameraTransform;
	};

	_NBL_STATIC_INLINE_CONSTEXPR uint8_t PATH_TO_MITSUBA_SCENE = 1;
	_NBL_STATIC_INLINE_CONSTEXPR uint8_t CAMERA_MATRIX_VALUES = 9;

	switch (rawVariables[id][DTEA_CAMERA_TRANSFORM].value().size())
	{
		case PATH_TO_MITSUBA_SCENE:
		{
			return getMatrixFromFile();
		}
		case CAMERA_MATRIX_VALUES:
		{
			return getMatrixFromSerializedValues();
		}
		default:
		{
			os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): " + std::string(CAMERA_TRANSFORM.data()) + " isn't a path nor a valid matrix! Id of input stride: " + std::to_string(id), ELL_ERROR);
			exit(-3);
		}
	}
}