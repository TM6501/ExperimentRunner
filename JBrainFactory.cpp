#include "pch.h"
#include "JBrainFactory.h"
#include "CGPFunctions.h"
#include "Enums.h"

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

extern unsigned int DEBUG_LEVEL;

namespace JBrain
{
	const float JBrainFactory::MIN_FLOAT_MUTATE_DIFF = 0.01f;
	const double JBrainFactory::MIN_DOUBLE_MUTATE_DIFF = 0.001;
	
	JBrainFactory* JBrainFactory::getInstance()
	{
		// Not built to handle the potential of multiple threads:
		static JBrainFactory* instance = new JBrainFactory();
		return instance;
	}

	JBrainFactory::JBrainFactory() :
		m_fullConfig(YAML::Null),	  
		m_dendriteConfig(YAML::Null),
		m_axonConfig(YAML::Null),
		m_neuronConfig(YAML::Null),
		m_sleepConfig(YAML::Null),
		m_brainConfig(YAML::Null),
		m_equationsConfig(YAML::Null),
		m_initialized(false),
		m_currentBrainNumber(1),
		m_requiredFunctions(),
		m_mutableFunctions(),
		m_observationProcessor(nullptr)
	{}

	JBrainFactory::~JBrainFactory()
	{
		if (m_observationProcessor != nullptr)
		{
			delete m_observationProcessor;
			m_observationProcessor = nullptr;
		}	
	}

	bool JBrainFactory::initialize(const std::string& yamlFilename)
	{
		// Load the full yaml:
		YAML::Node fullConfig = YAML::LoadFile(yamlFilename);

		// Run a different initialize depending on the paradigm:
		std::string paradigm = getConfigAsString(fullConfig, "BrainParadigm", true);

		if (paradigm == "growth")
			return initialize_growth(fullConfig);
		else if (paradigm == "snap")
			return initialize_snap(fullConfig);
		else
		{
			std::cout << "Unrecognized paradigm: " << paradigm << std::endl;
			return false;
		}
	}

	bool JBrainFactory::initialize_snap(const YAML::Node& fullConfig)
	{
		bool goodConfig = false;

		// Just check that each section exists. The library doesn't need to be fool-proof
		// with only a single user:
		if (fullConfig["StepEvents"] && fullConfig["OutputEvents"] && fullConfig["RunEvents"])
			goodConfig = true;

		m_fullConfig = fullConfig;		
		m_initialized = true;

		getObservationProcessor();

		return goodConfig;
	}

	YAML::Node JBrainFactory::getExperimentConfig()
	{
		if (m_fullConfig.IsNull())
			return YAML::Load("null");
		else
			return m_fullConfig["Experiment"];
	}

	std::vector<namePathTuple> JBrainFactory::getAllDoubleMutationParameters_snap()
	{
		// Only create it once:
		static std::vector<namePathTuple> retVal{
			namePathTuple("OverallProbability", { "OverallProbability" }),
			namePathTuple("NeuronFireThreshold", { "NeuronFireThreshold" }),
			namePathTuple("DendriteWeightChange", { "DendriteWeightChange" }),
		  namePathTuple("DendriteMinimumWeight", { "MinimumDendriteWeight" }),
		  namePathTuple("DendriteMaximumWeight", { "MaximumDendriteWeight" }),
		  namePathTuple("DendriteStartingWeight", { "DendriteStartingWeight" }),
		  namePathTuple("StepCreateNeuronChance", { "StepEvents", "CreateProcessingNeuron", "StartingChance"}),
		  namePathTuple("StepCreateNeuron_BaseCountRatioMultiplier", { "StepEvents", "CreateProcessingNeuron", "BaseOverCountRatioMultiplier"}),
		  namePathTuple("StepCreateInputNeuronChance", { "StepEvents", "CreateInputNeuron", "StartingChance"}),
		  namePathTuple("StepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier",
			  { "StepEvents", "CreateInputNeuron", "ObservationSizeOverInputNeuronCountMultiplier"}),
			namePathTuple("StepDestroyNeuronChance", { "StepEvents", "DestroyProcessingNeuron", "StartingChance"}),
			namePathTuple("StepDestroyNeuron_CountBaseRatioMultiplier",
				{ "StepEvents", "DestroyProcessingNeuron", "CountOverBaseRatioMultiplier"}),
			namePathTuple("StepDestroyInputNeuronChance", { "StepEvents", "DestroyInputNeuron", "StartingChance"}),
			namePathTuple("StepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier",
				{ "StepEvents", "DestroyInputNeuron", "InputNeuronCountOverObservationSizeMultiplier"}),
			namePathTuple("RunCreateNeuronChance", { "RunEvents", "CreateProcessingNeuron", "StartingChance"}),
			namePathTuple("RunCreateNeuron_BaseCountRatioMultiplier",
				{ "RunEvents", "CreateProcessingNeuron", "BaseOverCountRatioMultiplier"}),
			namePathTuple("RunCreateInputNeuronChance", { "RunEvents", "CreateInputNeuron", "StartingChance"}),
			namePathTuple("RunCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier",
				{ "RunEvents", "CreateInputNeuron", "ObservationSizeOverInputNeuronCountMultiplier"}),
			namePathTuple("RunDestroyNeuronChance", { "RunEvents", "DestroyProcessingNeuron", "StartingChance"}),
			namePathTuple("RunDestroyNeuron_CountBaseRatioMultiplier",
				{ "RunEvents", "DestroyProcessingNeuron", "CountOverBaseRatioMultiplier"}),
			namePathTuple("RunDestroyInputNeuronChance", { "RunEvents", "DestroyInputNeuron", "StartingChance"}),
			namePathTuple("RunDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier",
				{ "RunEvents", "DestroyInputNeuron", "InputNeuronCountOverObservationSizeMultiplier"}),
			namePathTuple("OutputPositive_CascadeProbability",
				{ "OutputEvents", "OutputPositive", "CascadeProbability"}),
			namePathTuple("OutputPositive_InSequence_IncreaseDendriteWeight",
				{ "OutputEvents", "OutputPositive", "FiredInSequence_IncreaseDendriteWeightFromInput"}),
			namePathTuple("OutputPositive_NoConnection_InSequence_CreateConnection",
				{ "OutputEvents", "OutputPositive", "NoConnectionButFiredInSequence_CreateConnection"}),
			namePathTuple("OutputPositive_YesFire_UnusedInput_DecreaseWeight",
				{ "OutputEvents", "OutputPositive", "YesFire_UnusedInput_DecreaseWeight"}),
			namePathTuple("OutputPositive_YesFire_UnusedInput_BreakConnection",
				{ "OutputEvents", "OutputPositive",  "YesFire_UnusedInput_BreakConnection"}),
			namePathTuple("OutputNegative_CascadeProbability",
				{ "OutputEvents", "OutputNegative", "CascadeProbability"}),
			namePathTuple("OutputNegative_InSequence_DecreaseDendriteWeight",
				{ "OutputEvents", "OutputNegative", "FiredInSequence_DecreaseDendriteWeightFromInput"}),
			namePathTuple("OutputNegative_InSequence_BreakConnection",
				{ "OutputEvents", "OutputNegative", "FiredInSequence_BreakConnection"}),
			namePathTuple("NoOutput_IncreaseInputDendriteWeight",
				{ "NoOutputEvents", "IncreaseInputDendriteWeight" }),
			namePathTuple("NoOutput_AddProcessingNeuronDendrite",
				{ "NoOutputEvents", "AddProcessingNeuronDendrite" }),
			namePathTuple("NoOutput_IncreaseProcessingNeuronDendriteWeight",
				{ "NoOutputEvents", "IncreaseProcessingNeuronDendriteWeight" }),
			namePathTuple("NoOutput_AddOutputNeuronDendrite", { "NoOutputEvents", "AddOutputNeuronDendrite" }),
			namePathTuple("NoOutput_IncreaseOutputNeuronDendriteWeight", { "NoOutputEvents", "IncreaseOutputNeuronDendriteWeight" }),
			namePathTuple("NoOutput_CreateProcessingNeuron", { "NoOutputEvents", "CreateProcessingNeuron" }) };
		return retVal;
	}

	std::vector<namePathTuple> JBrainFactory::getAllUIntMutationParameters_snap()
	{
		static std::vector<namePathTuple> retVal{
			namePathTuple("NeuronAccumulationDuration", { "NeuronAccumulateDuration" }),
			namePathTuple("BrainProcessingStepsAllowed", { "ProcessingStepsAllowed" }),
			namePathTuple("InitialInputNeuronCount", { "StartingInputNeuronCount" }),
			namePathTuple("InitialProcessingNeuronCount", { "StartingProcessingNeuronCount" }),
			namePathTuple("DendriteMinCountPerNeuron", { "MinimumDendritesPerNeuron" }),
			namePathTuple("DendriteMaxCountPerNeuron", { "MaximumDendritesPerNeuron" }),
			namePathTuple("DendriteStartCountPerNeuron", { "StartingDendritesPerNeuron" }),
			namePathTuple("BaseProcessingNeuronCount", { "BaseProcessingNeuronCount" }),
			// namePathTuple("ObservationSize", { "" }), // Shouldn't be mutated.
			// namePathTuple("ActionSize", { "" }) // Shouldn't be mutated.
		};
		return retVal;
	}

	std::vector<namePathTuple> JBrainFactory::getAllBoolMutationParameters_snap()
	{
		static std::vector<namePathTuple> retVal{
			namePathTuple("NeuronResetOnFiring", { "NeuronResetInputOnFiring" }),		
			namePathTuple("NeuronResetAfterOutput", { "NeuronFiresResetAfterOutput" }),		
			namePathTuple("DestroyNeuron_FavorFewerConnections", 
				{ "WeightDestroyProcessingNeuron", "FavorNeuronsWithFewerConnections" }),
			namePathTuple("DestroyNeuron_FavorYoungerNeurons",
				{ "WeightDestroyProcessingNeuron", "FavorYoungerNeurons" })			
		};
		return retVal;
	}

	std::vector<namePathTuple> JBrainFactory::getAllStringListMutationParameters_snap()
	{
		std::vector<namePathTuple> retVal{};

		retVal.push_back(namePathTuple("DynamicProbabilityUsage", { "DynamicProbabilityApplication" }));

		return retVal;
	}

	double JBrainFactory::getRandomListConfigAsDouble(const std::vector<std::string>& fullPath)
	{
		double minVal, maxVal;
		double retVal = -1.0;

		if (!getMinMaxDoubleFromConfig(minVal, maxVal, fullPath))
		{
			std::cout << "Failed to retrieve value as double: ";
			for (const auto& pathPart : fullPath)
				std::cout << pathPart << " ";
			std::cout << std::endl;
			return retVal;
		}

		// Got the min and max values, choose a random variable between them:
		return getRandomDouble(minVal, maxVal);
	}

	int JBrainFactory::getRandomListConfigAsInt(const std::vector<std::string>& fullPath)
	{
		int minVal, maxVal;
		int retVal = -1;

		if (!getMinMaxIntFromConfig(minVal, maxVal, fullPath))
		{
			std::cout << "Failed to retrieve value as int: ";
			for (const auto& pathPart : fullPath)
				std::cout << pathPart << " ";
			std::cout << std::endl;
			return retVal;
		}

		// Got the min and max values, choose a random variable between them:
		return getRandomInt(minVal, maxVal);
	}

	bool JBrainFactory::getRandomConfigAsBool(const std::vector<std::string>& fullPath)
	{
		std::string boolString = getValueAsString(fullPath);
		bool retBool = true;		
		if (boolString == "mutable")
			retBool = getRandomBool();
		else if (boolString == "false")
			retBool = false;
		// else
		//    retBool remains true;

		return retBool;
	}

	CGP::DYNAMIC_PROBABILITY JBrainFactory::getRandomDynamicProbabilityApplication()
	{
		std::string dynName = getConfigStringFromListOfStrings({ "DynamicProbabilityApplication" });
		return CGP::StringToDynamicProbability(dynName);
	}

	std::string JBrainFactory::getConfigStringFromListOfStrings(const std::vector<std::string>& fullPath)
	{
		// Allocate random devices only once:
		static std::mt19937_64 gen(std::random_device{}());
		
		bool foundFullPath = true;
		std::string retVal = "ERROR";
		
		// This copy-in, copy-out functionality makes this class not thread safe:
		YAML::Node original = YAML::Clone(m_fullConfig);
		YAML::Node nodeToCheck = m_fullConfig;

		// Move through sub nodes:
		for (auto configName : fullPath)
		{
			if (!nodeToCheck[configName])
			{
				foundFullPath = false;
				break;
			}
			nodeToCheck = nodeToCheck[configName];
		}

		if (foundFullPath)
		{
			std::vector<std::string> toSearch = nodeToCheck.as<std::vector<std::string> >();
			
			// Choose one randomly:
			std::uniform_int_distribution<> dist(0, static_cast<int>(toSearch.size()) - 1);
			unsigned int idx = static_cast<unsigned int>(dist(gen));
			retVal = toSearch[idx];			
		}

		// Put the original back in place and return success:
		m_fullConfig = original;
		return retVal;
	}

	std::string JBrainFactory::getValueAsString(const std::vector<std::string>& fullPath,
		bool convertToLowercase)
	{
		std::string retVal = "ERROR";
		bool foundFullPath = true;

		// This copy-in, copy-out functionality makes this class not thread safe:
		YAML::Node original = YAML::Clone(m_fullConfig);
		YAML::Node nodeToCheck = m_fullConfig;

		// Move through sub nodes:
		for (auto configName : fullPath)
		{
			if (!nodeToCheck[configName])
			{
				foundFullPath = false;
				break;
			}
			nodeToCheck = nodeToCheck[configName];
		}

		if (foundFullPath)
		{
			retVal = nodeToCheck.as<std::string>();			
		}

		if (convertToLowercase)
		{
			std::transform(retVal.begin(), retVal.end(), retVal.begin(),
				[](unsigned char c) { return std::tolower(c); });
		}

		// Put the original back in place and return success:
		m_fullConfig = original;
		return retVal;
	}

	bool JBrainFactory::getDoubleConfigDifferentValues(const std::vector<std::string> path)
	{
		double minVal, maxVal;
		bool retVal = false;

		if (!getMinMaxDoubleFromConfig(minVal, maxVal, path))
		{
			if (maxVal - minVal >= MIN_DOUBLE_MUTATE_DIFF)
				retVal = true;
		}

		return retVal;
	}

	bool JBrainFactory::getIntConfigDifferentValues(const std::vector<std::string> path)
	{
		int minVal, maxVal;
		bool retVal = false;

		if (!getMinMaxIntFromConfig(minVal, maxVal, path))
		{
			if (maxVal - minVal >= MIN_INT_MUTATE_DIFF)
				retVal = true;
		}

		return retVal;
	}

	bool JBrainFactory::getMinMaxDoubleFromConfig(double& outMin, double& outMax, const std::vector<std::string> subConfigs)
	{
		bool foundFullPath = true;
		outMin = -1.0;
		outMax = -1.0;
		// This copy-in, copy-out functionality makes this class not thread safe:
		YAML::Node original = YAML::Clone(m_fullConfig);
		YAML::Node nodeToCheck = m_fullConfig;

		// Move through sub nodes:
		for (auto configName : subConfigs)
		{
			if (!nodeToCheck[configName])
			{
				foundFullPath = false;
				break;
			}
			nodeToCheck = nodeToCheck[configName];
		}

		if (foundFullPath)
		{
			// Only 2 values, no need for loop:			
			outMin = nodeToCheck[0].as<double>();			
			outMax = nodeToCheck[1].as<double>();
		}

		// Put the original back in place and return success:
		m_fullConfig = original;
		return foundFullPath;
	}

	bool JBrainFactory::getMinMaxIntFromConfig(int& outMin, int& outMax, const std::vector<std::string> subConfigs)
	{
		bool foundFullPath = true;
		outMin = -1;
		outMax = -1;
		YAML::Node original = YAML::Clone(m_fullConfig);
		YAML::Node nodeToCheck = m_fullConfig;

		// Move through sub nodes:
		for (auto configName : subConfigs)
		{
			if (!nodeToCheck[configName])
			{
				foundFullPath = false;
				break;
			}
			nodeToCheck = nodeToCheck[configName];
		}

		if (foundFullPath)
		{
			// Only 2 values, no need for loop:			
			outMin = nodeToCheck[0].as<int>();
			outMax = nodeToCheck[1].as<int>();
		}

		// Put the original back in place and return success:
		m_fullConfig = original;
		return foundFullPath;
	}

	bool JBrainFactory::initialize_growth(const YAML::Node& fullConfig)
	{
		bool goodConfig = true;		
		
		// Check each individual section:
		if (fullConfig["Dendrite"])
		{
			// Check is always first so that the && doesn't short circuit.
			// The checks spit out errors and we want to see all of them in a single run.
			goodConfig = checkDendriteConfig(fullConfig["Dendrite"]) && goodConfig;
		}
		else
		{
			std::cout << "YAML configuration must include a 'Dendrite' section." << std::endl;
			goodConfig = false;
		}

		// Axon:
		if (fullConfig["Axon"])
			goodConfig = checkAxonConfig(fullConfig["Axon"]) && goodConfig;
		else
		{
			std::cout << "YAML configuration must include an 'Axon' section" << std::endl;
			goodConfig = false;
		}

		// Neuron:
		if (fullConfig["Neuron"])
			goodConfig = checkNeuronConfig(fullConfig["Neuron"]) && goodConfig;
		else
		{
			std::cout << "YAML configuration must include an 'Neuron' section" << std::endl;
			goodConfig = false;
		}

		// Sleep:
		if (fullConfig["Sleep"])
			goodConfig = checkSleepConfig(fullConfig["Sleep"]) && goodConfig;
		else
		{
			std::cout << "YAML configuration must include a 'Sleep' section" << std::endl;
			goodConfig = false;
		}

		// Brain:
		if (fullConfig["Brain"])
			goodConfig = checkBrainConfig(fullConfig["Brain"]) && goodConfig;
		else
		{
			std::cout << "YAML configuration must include a 'Brain' section" << std::endl;
			goodConfig = false;
		}
		
		// Equations:
		if (fullConfig["Equation"])
			goodConfig = checkEquationsConfig(fullConfig["Equation"]) && goodConfig;
		else
		{
			std::cout << "YAML configuration must include an 'Equation' section" << std::endl;
			goodConfig = false;
		}

		// Fill in the functions-available-to-CGP vectors. This call counts
		// on m_equationsConfig being valid:
		buildFunctionLists();

		m_initialized = true;
		return goodConfig;
	}

	bool JBrainFactory::checkDendriteConfig(const YAML::Node& config)
	{
		m_dendriteConfig = config;
		return true;
	}

	bool JBrainFactory::checkAxonConfig(const YAML::Node& config)
	{
		m_axonConfig = config;
		return true;	
	}

	bool JBrainFactory::checkNeuronConfig(const YAML::Node& config)
	{
		m_neuronConfig = config;

		m_neuronActivationFunctions.clear();
		for (unsigned int i = 0; i < m_neuronConfig["ActivationFunctions"].size(); ++i)
		{
			m_neuronActivationFunctions.push_back(
				CGP::StringToActivationFunction(
					m_neuronConfig["ActivationFunctions"][i].as<std::string>()));
		}

		std::string filename = getConfigAsString(m_neuronConfig, "NeuronDefinitionFile", false);

		// Should already be set as such, just emphasizing:
		m_staticNeuronsDefined = false;
		m_staticOutputNeuronsDefined = false;

		if (filename != "None" && filename != "none")
		{
			std::ifstream inFile(filename.c_str());
			if (inFile.is_open())
			{
				json j = json::parse(inFile);
				if (j.contains("neurons"))
				{
					m_staticNeuronsDefined = true;
					m_staticNeuronsJson = j["neurons"];
				}

				if (j.contains("outputNeurons"))
				{
					m_staticOutputNeuronsDefined = true;
					m_staticOutputNeuronsJson = j["outputNeurons"];
				}
			}
			else
			{
				std::cout << "Couldn't open \"" << filename << "\" as a json neuron definition file." << std::endl;
			}

			inFile.close();
		}

		return true;
	}

	bool JBrainFactory::checkSleepConfig(const YAML::Node& config)
	{
		m_sleepConfig = config;
		return true;
	}

	bool JBrainFactory::checkBrainConfig(const YAML::Node& config)
	{
		m_brainConfig = config;
		return true;
	}

	bool JBrainFactory::checkEquationsConfig(const YAML::Node& config)
	{
		m_equationsConfig = config;

		// Build the lists of inputs and outputs into the equations that
		// govern how the neurons change themselves:
		buildDendriteCGPLists();
		buildAxonCGPLists();
		buildNeuronCGPLists();

		return true;
	}

	void JBrainFactory::buildFunctionLists()
	{
		// Static variable of the functions to check:
		static std::vector<std::string> allFun{ "AND", "OR", "NAND", "NOR",
		  "XOR",  "ANDNOTY", "ADD", "SUBTRACT", "CMULT", "MULT", "DIVIDE",
		  "CDIVIDE", "INV", "ABS", "SQRTX", "SQRTXY", "CPOW", "POW", "EXPX",
		  "SINX", "ASINX", "COSX", "ACOSX", "TANX", "ATANX", "LTE", "GTE",
		  "LTEP", "GTEP", "MAX", "MIN", "XWIRE", "YWIRE", "CONST" };

		// Make sure they're clear at the start:
		m_requiredFunctions.clear();
		m_mutableFunctions.clear();

		// For every function, add it to the applicable list:
		std::string tmpStr;
		for (auto iter = std::begin(allFun); iter != std::end(allFun); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["AvailableFunctions"], *iter);
			if (tmpStr == "mutable")
				m_mutableFunctions.push_back(*iter);
			else if (tmpStr == "true")
				m_requiredFunctions.push_back(*iter);
			// else:
			    // false, do nothing.
		}
	}

	void JBrainFactory::buildDendriteCGPLists()
	{
		// Static list of inputs/outputs to check:
		static std::vector<std::string> allInputs{ "SAGE_MATCH_PERCENT",
		  "CURRENT_WEIGHT", "STRONGEST_INPUT_XYZ", "STRONGEST_INPUT_DISTANCE",
		  "STRONGEST_INPUT_IS_OBSERVATION_AXON", "STRONGEST_INPUT_VALUE",
		  "NEAREST_AXON_XYZ", "NEAREST_AXON_DISTANCE", "NEAREST_AXON_IS_OBSERVATION_AXON",
		  "INPUT_MAGNITUDE", "NEAREST_AXON_IS_PART_OF_SAME_NEURON", "CURRENT_LENGTH",
		  "NEURON_AGE", "NEURON_HEALTH", "EXPECTED_OUTPUT_DIFF"};

		static std::vector<std::string> allOutputs{"LOCATION", "HEALTH", "WEIGHT",
		"STRONGEST_INPUT_CLOSER_FURTHER", "NEAREST_AXON_CLOSER_FURTHER",
		"RANDOM_MOVEMENT_THRESHOLD", "CLOSER_TO_STRONGEST_INPUT",
		"CLOSER_TO_NEAREST_AXON"};
		
		m_dendriteInputs.clear();
		m_dendriteOutputs.clear();
		m_outputDendriteInputs.clear();
		m_outputDendriteOutputs.clear();

		// For each input, check to see if it is True:
		std::string tmpStr;
		for (auto iter = std::begin(allInputs); iter != std::end(allInputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["DendriteProgramInputs"], *iter);
			if (tmpStr == "true")
				m_dendriteInputs.push_back(CGP::StringToCGPInput(*iter));
		}

		// For each output, check to see if it is True:
		for (auto iter = std::begin(allOutputs); iter != std::end(allOutputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["DendriteProgramOutputs"], *iter);
			if (tmpStr == "true")
				m_dendriteOutputs.push_back(CGP::StringToCGPOutput(*iter));
		}

		// Repeat for the output dendrites:
		for (auto iter = std::begin(allInputs); iter != std::end(allInputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["OutputNeuronDendriteProgramInputs"], *iter);
			if (tmpStr == "true")
				m_outputDendriteInputs.push_back(CGP::StringToCGPInput(*iter));
		}

		// For each output, check to see if it is True:
		for (auto iter = std::begin(allOutputs); iter != std::end(allOutputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["OutputNeuronDendriteProgramOutputs"], *iter);
			if (tmpStr == "true")
				m_outputDendriteOutputs.push_back(CGP::StringToCGPOutput(*iter));
		}

		// Check for likely errors:
		if (std::find(m_dendriteInputs.begin(), m_dendriteInputs.end(),
			CGP::CGP_INPUT::UNDEFINED) != m_dendriteInputs.end())
		{
			std::cout << "Undefined found in dendrite inputs. Likely error." << std::endl;
		}

		if (std::find(m_dendriteOutputs.begin(), m_dendriteOutputs.end(),
			CGP::CGP_OUTPUT::UNDEFINED) != m_dendriteOutputs.end())
		{
			std::cout << "Undefined found in dendrite outputs. Likely error." << std::endl;
		}

		// Repeat check with output neurons:
		if (std::find(m_outputDendriteInputs.begin(), m_outputDendriteInputs.end(),
			CGP::CGP_INPUT::UNDEFINED) != m_outputDendriteInputs.end())
		{
			std::cout << "Undefined found in output dendrite inputs. Likely error." << std::endl;
		}

		if (std::find(m_outputDendriteOutputs.begin(), m_outputDendriteOutputs.end(),
			CGP::CGP_OUTPUT::UNDEFINED) != m_outputDendriteOutputs.end())
		{
			std::cout << "Undefined found in output dendrite outputs. Likely error." << std::endl;
		}
	}

	void JBrainFactory::buildAxonCGPLists()
	{
		// Static list of input & output values to check:
		static std::vector<std::string> allInputs{ "SAGE_MATCH_PERCENT",
		  "NEAREST_DENDRITE_XYZ", "NEAREST_DENDRITE_IS_PART_OF_SAME_NEURON",
		  "DENDRITE_TYPE", "PERCENTAGE_FIRE", "PERCENTAGE_BRAIN_FIRE",
		  "NEURON_AGE", "NEURON_HEALTH", "CURRENT_LENGTH"};

		static std::vector<std::string> allOutputs{ "LOCATION",
			"NEAREST_DENDRITE_CLOSER_FURTHER", "RANDOM_MOVEMENT_THRESHOLD",
			"HEALTH" };

		m_axonInputs.clear();
		m_axonOutputs.clear();

		// For each, check to see if it is True:
		std::string tmpStr;
		for (auto iter = std::begin(allInputs); iter != std::end(allInputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["AxonProgramInputs"], *iter);
			if (tmpStr == "true")
				m_axonInputs.push_back(CGP::StringToCGPInput(*iter));
		}

		// For each output, check to see if it is True:
		for (auto iter = std::begin(allOutputs); iter != std::end(allOutputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["AxonProgramOutputs"], *iter);
			if (tmpStr == "true")
				m_axonOutputs.push_back(CGP::StringToCGPOutput(*iter));
		}

		// Check for likely errors:
		if (std::find(m_axonInputs.begin(), m_axonInputs.end(),
			CGP::CGP_INPUT::UNDEFINED) != m_axonInputs.end())
		{
			std::cout << "Undefined found in axon inputs. Likely error." << std::endl;
		}

		if (std::find(m_axonOutputs.begin(), m_axonOutputs.end(),
			CGP::CGP_OUTPUT::UNDEFINED) != m_axonOutputs.end())
		{
			std::cout << "Undefined found in axon outputs. Likely error." << std::endl;
		}
	}

	void JBrainFactory::buildNeuronCGPLists()
	{
		// Static list of input and values to check:
		static std::vector<std::string> allInputs{"SAGE_MATCH_PERCENT",
		  "NEURON_HEALTH", "PERCENTAGE_FIRE", "PERCENTAGE_BRAIN_FIRE",
		  "NEURON_AGE", "NEURON_HEALTH" };

		static std::vector<std::string> allOutputs{"HEALTH", "HEALTH_INCREASE_DECREASE"};

		m_neuronInputs.clear();
		m_neuronOutputs.clear();

		// For each, check to see if it is True:
		std::string tmpStr;
		for (auto iter = std::begin(allInputs); iter != std::end(allInputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["NeuronProgramInputs"], *iter);
			if (tmpStr == "true")
				m_neuronInputs.push_back(CGP::StringToCGPInput(*iter));
		}

		// For each output, check to see if it is True:
		for (auto iter = std::begin(allOutputs); iter != std::end(allOutputs); ++iter)
		{
			tmpStr = getConfigAsString(m_equationsConfig["NeuronProgramOutputs"], *iter);
			if (tmpStr == "true")
				m_neuronOutputs.push_back(CGP::StringToCGPOutput(*iter));
		}
	}

	void JBrainFactory::getRandomCGPFunctionLists(
	  std::vector<std::string>& functionNames,
	  std::vector<std::function<double(double, double, double)> >& functions)
	{
		functionNames = m_requiredFunctions;
		for (auto iter = std::begin(m_mutableFunctions); iter != std::end(m_mutableFunctions); ++iter)
		{
			// For every mutable function, flip a coin:
			if (getRandomBool())
				functionNames.push_back(*iter);
		}

		// For all of the gathered function names, add each function:
		functions.clear();
		for (auto iter = std::begin(functionNames); iter != std::end(functionNames); ++iter)
			functions.push_back(CGPFunctions::doubleIn_doubleOut::getFuncFromString(*iter));
	}

	std::string JBrainFactory::getNextBrainName()
	{
		// Brain names are "B<brain number>"
		std::string retVal = std::string("B") + std::to_string(m_currentBrainNumber);
		++m_currentBrainNumber;
		return retVal;
	}

	float JBrainFactory::getConfigAsFloat(const YAML::Node& config, const std::string& name)
	{
		if (!config[name])
		{
			std::cout << name << " is not a valid YAML node name." << std::endl;
			return -1.0;
		}
		else
			return config[name].as<float>();
	}

	int JBrainFactory::getConfigAsInt(const YAML::Node& config, const std::string& name)
	{
		if (!config[name])
		{
			std::cout << name << " is not a valid YAML node name." << std::endl;
			return -1;
		}
		return config[name].as<int>();
	}

	std::string JBrainFactory::getConfigAsString(const YAML::Node& config, const std::string& name, bool convertToLowercase)
	{
		if (!config[name])
		{
			std::cout << name << " is not a valid YAML node name." << std::endl;
			return "";
		}

	    std::string retString = config[name].as<std::string>();
		if (convertToLowercase)
		{
			std::transform(retString.begin(), retString.end(), retString.begin(),
				[](unsigned char c) { return std::tolower(c); });
		}

		return retString;
	}

	bool JBrainFactory::getConfigAsMutableBool(const YAML::Node& config, const std::string& name)
	{
		bool retBool = true;
		const std::string& configString = getConfigAsString(config, name);
		if (configString == "mutable")
			retBool = getRandomBool();
		else if (configString == "false")
			retBool = false;
		// else
		//    retBool remains true;
		
		return retBool;
	}

	float JBrainFactory::getRandomFloat(const float& min, const float& max)
	{
		// Random device and distribution don't need to be
		// recreated every time:
		static std::random_device rd;
		static std::mt19937 e2(rd());
		
		// We always want uniform distribution. The odd next-after
		// syntax around max is used to make sure that max is one
		// of the values that can be returned. The distribution's possible
		// return values are in the range [a, b):
		std::uniform_real_distribution<> dist(min,
		  std::nextafter(max, std::numeric_limits<float>::max()));

		return static_cast<float>(dist(e2));
	}

	double JBrainFactory::getRandomDouble(const double& min, const double& max)
	{
		// Random device and distribution don't need to be
		// recreated every time:
		static std::random_device rd;
		static std::mt19937_64 e2(rd());

		// We always want uniform distribution. The odd next-after
		// syntax around max is used to make sure that max is one
		// of the values that can be returned. The distribution's possible
		// return values are in the range [a, b):
		std::uniform_real_distribution<> dist(min,
			std::nextafter(max, std::numeric_limits<double>::max()));

		return dist(e2);
	}

	float JBrainFactory::getFloatFromConfigRange(const YAML::Node& config, const std::string& minName, const std::string& maxName)
	{
		return getRandomFloat(
			getConfigAsFloat(config, minName),
			getConfigAsFloat(config, maxName)
		);		
	}

	int JBrainFactory::getIntFromConfigRange(const YAML::Node& config, const std::string& minName, const std::string& maxName)
	{
		return getRandomInt(
			getConfigAsInt(config, minName),
			getConfigAsInt(config, maxName)
		);
	}

	CGP::JNEURON_ACTIVATION_FUNCTION JBrainFactory::getRandomJNeuronActivationFunction()
	{
		CGP::JNEURON_ACTIVATION_FUNCTION retVal = m_neuronActivationFunctions[0];
		if (m_neuronActivationFunctions.size() > 1)
		{
			int idx = getRandomInt(0, static_cast<int>(m_neuronActivationFunctions.size() - 1));
			retVal = m_neuronActivationFunctions[idx];
		}

		return retVal;
	}

	int JBrainFactory::getRandomInt(const int& min, const int& max)
	{
		// Random device and distribution don't need to be
	  // recreated every time:
		static std::random_device rd;
		static std::mt19937 e2(rd());
				
		std::uniform_int_distribution<> dist(min, max);

		return dist(e2);
	}

	bool JBrainFactory::getRandomBool()
	{
		// Create static versions of our own random devices/distributions;
		// it is faster than using the getRandomInt function:
		static std::random_device rd;
		static std::mt19937 e2(rd());
		static std::uniform_int_distribution<> dist(0, 2);

		return bool(dist(e2) == 0);
	}

	ObservationProcessor* JBrainFactory::getObservationProcessor()
	{
		// If we don't have an observation processor yet, create it:
		if (m_observationProcessor == nullptr)
		{
			CGP::INPUT_PREPROCESSING preProc = CGP::StringToInputPreprocessing(
				getConfigAsString(m_fullConfig["InputProcessing"], "Type", false));
			unsigned int obsSize = static_cast<unsigned int>(
				getConfigAsInt(m_fullConfig["InputProcessing"], "ObservationSize"));
			unsigned int buckets = static_cast<unsigned int>(
				getConfigAsInt(m_fullConfig["InputProcessing"], "BucketsPerInput"));
			std::vector<std::vector<double> > obsRanges;
			for (auto innerList : m_fullConfig["InputProcessing"]["ObsRanges"])
			{
				// Should always be of length 2:
				obsRanges.push_back(std::vector<double>{ innerList[0].as<double>(), innerList[1].as<double>() });
			}

			m_observationProcessor = new ObservationProcessor(preProc, obsSize, obsRanges, buckets);
		}

		return m_observationProcessor;
	}

	JBrain_Snap* JBrainFactory::getRandomSnapBrain()
	{
		if (!m_initialized)
		{
			std::cout << "Asking for a snap brain from uninitialized factory." << std::endl;
			return nullptr;
		}

		ObservationProcessor* obsProc = getObservationProcessor();

		JBrain_Snap* retVal = new JBrain_Snap(
			getNextBrainName(), // name
			std::string("JBrainFactory"), // parentName
			getRandomListConfigAsDouble({ "OverallProbability" }), // overallProbability
			getRandomDynamicProbabilityApplication(),  // dynamicProbabilityUsage
			getRandomListConfigAsDouble({ "DynamicProbabilityMultiplier" }), // dynamicProbabilityMultiplier
			static_cast<unsigned int>(getRandomListConfigAsInt({ "NeuronAccumulateDuration" })), // neuronAccumulateDuration
			getRandomConfigAsBool({ "NeuronResetInputOnFiring" }), // neuronResetOnFiring
			getRandomConfigAsBool({ "NeuronFiresResetAfterOutput" }), // neuronResetAfterOutput
			getRandomListConfigAsDouble({ "NeuronFireThreshold" }), // neuronFireThreshold
			static_cast<unsigned int>(getRandomListConfigAsInt({ "NeuronMaximumAge" })),
			static_cast<unsigned int>(getRandomListConfigAsInt({ "ProcessingStepsAllowed" })), //brainProcessingStepsAllowed
			getRandomListConfigAsDouble({ "DendriteWeightChange" }), // dendriteWeightChange
			getRandomListConfigAsDouble({ "MinimumDendriteWeight" }), // dendriteMinimumWeight
			getRandomListConfigAsDouble({ "MaximumDendriteWeight" }), // dendriteMaximumWeight
			getRandomListConfigAsDouble({ "DendriteStartingWeight" }), // dendriteStartingWeight
			static_cast<unsigned int>(getRandomListConfigAsInt({ "MinimumDendritesPerNeuron" })), // dendriteMinCountPerNeuron
			static_cast<unsigned int>(getRandomListConfigAsInt({ "MaximumDendritesPerNeuron" })), // dendriteMaxCountPerNeuron
			static_cast<unsigned int>(getRandomListConfigAsInt({ "StartingDendritesPerNeuron" })), // dendriteStartCountPerNeuron
			static_cast<unsigned int>(getRandomListConfigAsInt({ "BaseProcessingNeuronCount" })), // baseProcessingNeuronCount
			static_cast<unsigned int>(getRandomListConfigAsInt({ "EnvironmentDetails", "ActionSize" })), // actionSize,
			static_cast<unsigned int>(getRandomListConfigAsInt({ "StartingInputNeuronCount" })), // initialInputNeuronCount
			static_cast<unsigned int>(getRandomListConfigAsInt({ "StartingProcessingNeuronCount" })), // initialProcessingNeuronCount
			static_cast<unsigned int>(getRandomListConfigAsInt({ "MaximumProcessingNeuronCount" })), // maximumProcessingNeuronCount
			static_cast<unsigned int>(getRandomListConfigAsInt({ "MaximumInputNeuronToInputsRatio" })), // maximumInputNeuronToInputsRatio
			getRandomListConfigAsDouble({ "StepEvents", "CreateProcessingNeuron", "StartingChance" }), //stepCreateNeuronChance,
			getRandomListConfigAsDouble({ "StepEvents", "CreateProcessingNeuron", "BaseOverCountRatioMultiplier" }), //stepCreateNeuron_BaseCountRatioMultiplier,
			getRandomListConfigAsDouble({ "StepEvents", "CreateInputNeuron", "StartingChance" }), //stepCreateInputNeuronChance,
			getRandomListConfigAsDouble({ "StepEvents", "CreateInputNeuron", "ObservationSizeOverInputNeuronCountMultiplier" }), //stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			getRandomListConfigAsDouble({ "StepEvents", "DestroyProcessingNeuron", "StartingChance" }), //stepDestroyNeuronChance,
			getRandomListConfigAsDouble({ "StepEvents", "DestroyProcessingNeuron", "CountOverBaseRatioMultiplier" }), //stepDestroyNeuron_CountBaseRatioMultiplier,
			getRandomListConfigAsDouble({ "StepEvents", "DestroyInputNeuron", "StartingChance" }), //stepDestroyInputNeuronChance,
			getRandomListConfigAsDouble({ "StepEvents", "DestroyInputNeuron", "InputNeuronCountOverObservationSizeMultiplier" }), //stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier
			getRandomConfigAsBool({ "WeightDestroyProcessingNeuron", "FavorNeuronsWithFewerConnections" }), //DestroyNeuron_FavorFewerConnections
			getRandomConfigAsBool({ "WeightDestroyProcessingNeuron", "FavorYoungerNeurons" }), // DestroyNeuron_favorYoungerNeurons
			getRandomListConfigAsDouble({ "RunEvents", "CreateProcessingNeuron", "StartingChance" }),  //runCreateNeuronChance,
			getRandomListConfigAsDouble({ "RunEvents", "CreateProcessingNeuron", "BaseOverCountRatioMultiplier" }),  //runCreateNeuron_BaseCountRatioMultiplier,
			getRandomListConfigAsDouble({ "RunEvents", "CreateInputNeuron", "StartingChance" }),  //runCreateInputNeuronChance,
			getRandomListConfigAsDouble({ "RunEvents", "CreateInputNeuron", "ObservationSizeOverInputNeuronCountMultiplier" }),  //runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			getRandomListConfigAsDouble({ "RunEvents", "DestroyProcessingNeuron", "StartingChance" }),  //runDestroyNeuronChance,
			getRandomListConfigAsDouble({ "RunEvents", "DestroyProcessingNeuron", "CountOverBaseRatioMultiplier" }),  //runDestroyNeuron_CountBaseRatioMultiplier,
			getRandomListConfigAsDouble({ "RunEvents", "DestroyInputNeuron", "StartingChance" }),  //runDestroyInputNeuronChance,
			getRandomListConfigAsDouble({ "RunEvents", "DestroyInputNeuron", "InputNeuronCountOverObservationSizeMultiplier" }),  //runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputPositive", "CascadeProbability" }),  //outputPositive_CascadeProbability,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputPositive", "FiredInSequence_IncreaseDendriteWeightFromInput" }),  //outputPositive_InSequence_IncreaseDendriteWeight,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputPositive", "NoConnectionButFiredInSequence_CreateConnection" }),  //outputPositive_NoConnection_InSequence_CreateConnection,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputPositive", "YesFire_UnusedInput_DecreaseWeight" }),  //outputPositive_YesFire_UnusedInput_DecreaseWeight,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputPositive", "YesFire_UnusedInput_BreakConnection" }),  //outputPositive_YesFire_UnusedInput_BreakConnection,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputNegative", "CascadeProbability" }),  //outputNegative_CascadeProbability,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputNegative", "FiredInSequence_DecreaseDendriteWeightFromInput" }),  //outputNegative_InSequence_DecreaseDendriteWeight,
			getRandomListConfigAsDouble({ "OutputEvents", "OutputNegative", "FiredInSequence_BreakConnection" }),  //outputNegative_InSequence_BreakConnection,
			getRandomListConfigAsDouble({ "NoOutputEvents", "IncreaseInputDendriteWeight" }),  //NoOutput_IncreaseInputDendriteWeight,
			getRandomListConfigAsDouble({ "NoOutputEvents", "AddProcessingNeuronDendrite" }),  //NoOutput_AddProcessingNeuronDendrite,
			getRandomListConfigAsDouble({ "NoOutputEvents", "IncreaseProcessingNeuronDendriteWeight" }),  //NoOutput_IncreaseProcessingNeuronDendriteWeight,
			getRandomListConfigAsDouble({ "NoOutputEvents", "AddOutputNeuronDendrite" }),  //NoOutput_AddOutputNeuronDendrite,
			getRandomListConfigAsDouble({ "NoOutputEvents", "IncreaseOutputNeuronDendriteWeight" }),  //NoOutput_IncreaseOutputNeuronDendriteWeight,
			getRandomListConfigAsDouble({ "NoOutputEvents", "CreateProcessingNeuron" }),  //NoOutput_CreateProcessingNeuron,
			obsProc); // observation processor

		return retVal;
	}

	JBrain* JBrainFactory::getRandomBrain()
	{
		if (!m_initialized)
		{
			std::cout << "Asking for a brain from uninitialized factory." << std::endl;
			return nullptr;
		}		
				
		// brainUseSameDimensions
		bool brainUseSameDimensions = getConfigAsMutableBool(
			m_brainConfig, "UseSameValueForAllDimensions");
		
		// brainXSize / y / z
		float brainXSize = getRandomFloat(
			getConfigAsFloat(m_brainConfig, "MinDimension"),
			getConfigAsFloat(m_brainConfig, "MaxDimension")
		);

		// Assume same dimensions:
		float brainYSize = brainXSize;
		float brainZSize = brainXSize;

		// Not the same? Choose new random values:
		if (!brainUseSameDimensions)
		{
			brainYSize = getRandomFloat(
				getConfigAsFloat(m_brainConfig, "MinDimension"),
				getConfigAsFloat(m_brainConfig, "MaxDimension")
			);
			brainZSize = getRandomFloat(
				getConfigAsFloat(m_brainConfig, "MinDimension"),
				getConfigAsFloat(m_brainConfig, "MaxDimension")
			);
		}

		// Get valid function lists for the CGP individuals:
		std::vector<std::string> functionStringList;
		std::vector<std::function<double(double, double, double)> >functionList;
		getRandomCGPFunctionLists(functionStringList, functionList);
		bool useOutputNeurons = getConfigAsMutableBool(m_brainConfig, "UseOutputNeurons");

		JBrain* retBrain = new JBrain(	
			getNextBrainName(), // Name
			"JBrainFactory", // Parent's name.
			static_cast<unsigned int>(getConfigAsInt(m_brainConfig, "InputCount")), // observationSize
			static_cast<unsigned int>(getConfigAsInt(m_brainConfig, "OutputCount")),  // actionSize
			getFloatFromConfigRange(m_dendriteConfig, "MinMaxLength", "MaxMaxLength"), //dendriteMaxLength
			static_cast<unsigned int>(getIntFromConfigRange(m_dendriteConfig, "MinMinCount", "MaxMinCount")), // dendriteMinCount
			static_cast<unsigned int>(getIntFromConfigRange(m_dendriteConfig, "MinMaxCount", "MaxMaxCount")), // dendriteMaxCount
			getFloatFromConfigRange(m_dendriteConfig, "MinMinWeight", "MaxMinWeight"), // dendriteMinWeight
			getFloatFromConfigRange(m_dendriteConfig, "MinMaxWeight", "MaxMaxWeight"), // dendriteMaxWeight
			getFloatFromConfigRange(m_dendriteConfig, "MinLowMoveAway", "MaxLowMoveAway"), // dendriteLowMoveAway
			getFloatFromConfigRange(m_dendriteConfig, "MinHighMoveToward", "MaxHighMoveToward"), // dendriteHighMoveToward
			getFloatFromConfigRange(m_dendriteConfig, "MinMoveAmount", "MaxMoveAmount"), // dendriteAwayTowardMoveAmount
			getFloatFromConfigRange(m_dendriteConfig, "MinLowWeightDecrease", "MaxLowWeightDecrease"), // dendriteLowWeightDecrease
			getFloatFromConfigRange(m_dendriteConfig, "MinHighWeightIncrease", "MaxHighWeightIncrease"), // dendriteHighWeightIncrease
			getFloatFromConfigRange(m_dendriteConfig, "MinWeightChangeAmount", "MaxWeightChangeAmount"), // dendriteWeightChangeAmount
			getFloatFromConfigRange(m_axonConfig, "MinMaxLength", "MaxMaxLength"), // axonMaxLength
			static_cast<unsigned int>(getIntFromConfigRange(m_axonConfig, "MinMinCount", "MaxMinCount")), //axonMinCount
			static_cast<unsigned int>(getIntFromConfigRange(m_axonConfig, "MinMaxCount", "MaxMaxCount")), //axonMaxCount
			getFloatFromConfigRange(m_axonConfig, "MinLowMoveAway", "MaxLowMoveAway"), // axonLowMoveAway
			getFloatFromConfigRange(m_axonConfig, "MinHighMoveToward", "MaxHighMoveToward"), // axonHighMoveToward
			getFloatFromConfigRange(m_axonConfig, "MinMoveAmount", "MaxMoveAmount"), // axonAwayTowardMoveAmount
			getConfigAsMutableBool(m_neuronConfig, "FireProbabilistic"),  // neuronProbabilisticFire
			getFloatFromConfigRange(m_neuronConfig, "MinFireThreshold", "MaxFireThreshold"),  // neuronFireThreshold
            getFloatFromConfigRange(m_neuronConfig, "MinMinFireValue", "MaxMinFireValue"), // neuronMinFireValue
			getFloatFromConfigRange(m_neuronConfig, "MinMaxFireValue", "MaxMaxFireValue"), // neuronMaxFireValue
			getConfigAsMutableBool(m_neuronConfig, "UseDynamicFireThresholds"), // neuronUseDynamicFireThresholds
			getFloatFromConfigRange(m_neuronConfig, "MinFireThresholdIdleChange", "MaxFireThresholdIdleChange"), // neuronDynamicFireThresholdIdleChange
			getFloatFromConfigRange(m_neuronConfig, "MinFireThresholdActiveChange", "MaxFireThresholdActiveChange"), // neuronDynamicFireThresholdActiveChange
			static_cast<unsigned int>(getIntFromConfigRange(m_neuronConfig, "MinRefractoryPeriod", "MaxRefractoryPeriod")), //neuronRefractoryPeriod
			getConfigAsMutableBool(m_neuronConfig, "NeuronDuplicatesNearby"),  // neuronDuplicateNearby
			getFloatFromConfigRange(m_neuronConfig, "MinMinNearbyDistance", "MaxMinNearbyDistance"), //neuronMinNearbyDistance
			getFloatFromConfigRange(m_neuronConfig, "MinMaxNearbyDistance", "MaxMaxNearbyDistance"), //neuronMaxNearbyDistance
			static_cast<unsigned int>(getIntFromConfigRange(m_brainConfig, "MinMinStartingNeurons", "MaxMinStartingNeurons")), // minStartingNeurons
			static_cast<unsigned int>(getIntFromConfigRange(m_brainConfig, "MinMaxStartingNeurons", "MaxMaxStartingNeurons")), // maxStartingNeurons
			static_cast<unsigned int>(getConfigAsInt(m_brainConfig, "MaxNeuronCount")), // maxNeuronCount
			useOutputNeurons, // useOutputNeurons
			getFloatFromConfigRange(m_neuronConfig, "MinNeuronStartingHealth", "MaxNeuronStartingHealth"),
			getFloatFromConfigRange(m_neuronConfig, "MinLowHealthChange", "MaxLowHealthChange"),  // neuronCGPOutputLowHealthChange
			getFloatFromConfigRange(m_neuronConfig, "MinHighHealthChange", "MaxHighHealthChange"),  // neuronCGPOutputHighHealthChange
			getFloatFromConfigRange(m_neuronConfig, "MinHealthChangeAmount", "MaxHealthChangeAmount"), // neuronCGPOutputHealthChangeAmount
			getFloatFromConfigRange(m_neuronConfig, "MinStartingDeathHealth", "MaxStartingDeathHealth"), // neuronDeathHealth
			getFloatFromConfigRange(m_neuronConfig, "MinStartingDuplicateHealth", "MaxStartingDuplicateHealth"), //  neuronDulicateHealth
			getFloatFromConfigRange(m_neuronConfig, "MinHealthThresholdMultiplier", "MaxHealthThresholdMultiplier"), // neuronDeathDuplicateHealthThresholdMultiplier
			getFloatFromConfigRange(m_neuronConfig, "MinNeuronDuplicateHealthChange", "MaxNeuronDuplicateHealthChange"), // neuronDuplicationHealthChange 
			getConfigAsMutableBool(m_neuronConfig, "HealthResetAtDuplication"), // neuronDuplicationHealthReset
			getRandomJNeuronActivationFunction(), // neuronActivationFunction
			getFloatFromConfigRange(m_neuronConfig, "MinNeuronSpaceDeteriorationParameter", "MaxNeuronSpaceDeteriorationParameter"), // neuronFireSpaceDeterioration
			getFloatFromConfigRange(m_neuronConfig, "MinNeuronTimeDeteriorationParameter", "MaxNeuronTimeDeteriorationParameter"), // neuronFireTimeDeterioration
			static_cast<unsigned int>(getIntFromConfigRange(m_neuronConfig, "MinNeuronFireLifetime", "MaxNeuronFireLifetime")), //neuronFireLifetime
		    getConfigAsMutableBool(m_brainConfig, "InputsAge"), // inputNeuronFiresAge
			getConfigAsMutableBool(m_sleepConfig, "UsePreTrainSleep"),  //usePreTrainSleep
			getConfigAsMutableBool(m_sleepConfig, "UsePostTrainSleep"),  //usePostTrainSleep
			brainXSize, brainYSize, brainZSize, brainUseSameDimensions,  // brain dimensions
			getConfigAsMutableBool(m_brainConfig, "ResetBeforeProcessingInput"),  // brainResetBeforeProcessingInput
			static_cast<unsigned int>(getIntFromConfigRange(m_brainConfig, "MinProcessingTimeStepsBetweenInputAndOutput",
				"MaxProcessingTimeStepsBetweenInputAndOutput")),  // brainProcessingStepsBetweenInputAndOutput
			static_cast<unsigned int>(getIntFromConfigRange(m_brainConfig,
				"MinOutputTimeStepsToAverageTogether", "MaxOutputTimeStepsToAverageTogether")), //braitOutputsToAverageTogether
			getConfigAsMutableBool(m_brainConfig, "InputsOnOneSide"), // brainInputsOnOneSide
			getConfigAsMutableBool(m_brainConfig, "OutputsOnOneSide"), // brainOutputsOnOneSide
			getConfigAsMutableBool(m_brainConfig, "OutputsIgnoreEnvironmentInputs"), //brainOutputsIgnoreEnvironmentInputs
			getFloatFromConfigRange(m_equationsConfig, "MinMinP", "MaxMinP"), // MinP
			getFloatFromConfigRange(m_equationsConfig, "MinMaxP", "MaxMaxP"), // MaxP
			getFloatFromConfigRange(m_equationsConfig, "MinLowConstraint", "MaxLowConstraint"), // minConstraint
			getFloatFromConfigRange(m_equationsConfig, "MinHighConstraint", "MaxHighConstraint"), // maxConstraint
			static_cast<unsigned int>(getConfigAsInt(m_neuronConfig, "MaxNeuronAge")), //maxNeuronAge
			m_dendriteInputs, m_dendriteOutputs,
			static_cast<unsigned int>(getConfigAsInt(m_equationsConfig, "DendriteProgramCGPNodes")), // dendriteProgramNodes
			m_outputDendriteInputs, m_outputDendriteOutputs,
		    static_cast<unsigned int>(getConfigAsInt(m_equationsConfig, "OutputDendriteProgramCGPNodes")), // outputDendriteProgramNodes)
			m_axonInputs, m_axonOutputs,
			static_cast<unsigned int>(getConfigAsInt(m_equationsConfig, "AxonProgramCGPNodes")), // axonProgramNodes
			m_neuronInputs, m_neuronOutputs,
			static_cast<unsigned int>(getConfigAsInt(m_equationsConfig, "NeuronProgramCGPNodes")), // neuronProgramNodes
			CGP::StringToUpdateEvent(getConfigAsString(m_equationsConfig, "UpdateProgramsEvent")),
			static_cast<unsigned int>(getIntFromConfigRange(m_equationsConfig, "MinUpdateProgramsFrequency", "MaxUpdateProgramsFrequency")),
			functionStringList, functionList // Functions available to CGP
		);

		// Create either pre-defined or random neurons:
		if (m_staticNeuronsDefined)
			retBrain->setNeuronsFromStaticJson(m_staticNeuronsJson, false);
		else // Tell the brain to create enough random neurons:
			retBrain->addRandomStartingNeurons();

		// Output neurons, if defined, are already created. This will overwrite them:
		if (useOutputNeurons && m_staticOutputNeuronsDefined)
			retBrain->setNeuronsFromStaticJson(m_staticOutputNeuronsJson, true);

		// Still need to create all the details of the equations and have the brain create CGP programs for each or one for all.
		return retBrain;
	}

	JBrain* JBrainFactory::getMutatedBrain_float(JBrain* parent,
		const YAML::Node& config, const std::string& configValueName,
		const std::string& brainValueName)
	{
		// Min/max values are required to be named <Min/Max><ValueName>:
		std::string minValName = "Min" + configValueName;
		std::string maxValName = "Max" + configValueName;

		float minVal = getConfigAsFloat(config, minValName);
		float maxVal = getConfigAsFloat(config, maxValName);

		// Null return value indicates that no mutation was possible on
		// this value:
		JBrain* retVal = nullptr;

		if ((maxVal - minVal) > MIN_FLOAT_MUTATE_DIFF)
		{
			// Get the new mutated value:
			float mutVal = getRandomFloat(minVal, maxVal);

			retVal = new JBrain(*parent);
			if (!retVal->setValueByName(brainValueName, mutVal))
			{
				std::cout << "Failed to set value by name for: " << configValueName
					<< " -> " << brainValueName << std::endl;
			}
		}

		return retVal;
	}

	std::vector<std::string> JBrainFactory::getListOfStrings(std::vector<std::string> fullPath)
	{
		// This copy-in, copy-out functionality makes this class not thread safe:
		YAML::Node original = YAML::Clone(m_fullConfig);
		YAML::Node nodeToCheck = m_fullConfig;

		std::vector<std::string> retVal{};
		bool foundFullPath = true;

		// Move through sub nodes:
		for (auto configName : fullPath)
		{
			if (!nodeToCheck[configName])
			{
				foundFullPath = false;
				break;
			}
			nodeToCheck = nodeToCheck[configName];
		}

		if (foundFullPath)
			retVal = nodeToCheck.as<std::vector<std::string> >();

		m_fullConfig = original;
		return retVal;
	}

	JBrain_Snap* JBrainFactory::getStringMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param)
	{
		std::vector<std::string> options = getListOfStrings(std::get<1>(param));
		JBrain_Snap* retVal = nullptr;

		// If the length of the list is long enough, choose randomly.
		// This can result in no change in the brain in question.
		if (options.size() >= MIN_STRING_LIST_LENGTH)
		{
			int idx = getRandomInt(0, static_cast<int>(options.size() - 1));
			std::string mutVal = options[idx];
			
			retVal = new JBrain_Snap(*parent);
			if (!retVal->setValueByName(std::get<0>(param), mutVal))
			{
				std::cout << "Failed to set string value by name: " << std::get<0>(param) << std::endl;
				delete retVal;
				retVal = nullptr;
			}
			else
				retVal->setValueByName("Name", getNextBrainName());
		}

		return retVal;
	}

	JBrain_Snap* JBrainFactory::getDoubleMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param)
	{
		// Get the min and max values:
		double minVal, maxVal;
		getMinMaxDoubleFromConfig(minVal, maxVal, std::get<1>(param));
		JBrain_Snap* retVal = nullptr;

		// Mutate if there is a need:
		if ((maxVal - minVal) > MIN_DOUBLE_MUTATE_DIFF)
		{
			double mutVal = getRandomDouble(minVal, maxVal);
			retVal = new JBrain_Snap(*parent);

			if (!retVal->setValueByName(std::get<0>(param), mutVal))
			{
				std::cout << "Failed to set double value by name: " << std::get<0>(param) << std::endl;
				delete retVal;
				retVal = nullptr;
			}
			else
				retVal->setValueByName("Name", getNextBrainName());
		}

		return retVal;
	}

	JBrain* JBrainFactory::getMutatedBrain_int(JBrain* parent,
		const YAML::Node& config, const std::string& configValueName, const std::string& brainValueName)
	{
		// Min/max values are required to be named <Min/Max><ValueName>:
		std::string minValName = "Min" + configValueName;
		std::string maxValName = "Max" + configValueName;

		int minVal = getConfigAsInt(config, minValName);
		int maxVal = getConfigAsInt(config, maxValName);

		// Null return means we couldn't mutate this value:
		JBrain* retVal = nullptr;

		if ((maxVal - minVal) >= MIN_INT_MUTATE_DIFF)
		{
			// For small mutation ranges, this may end up not modifying the value:
			int mutVal = getRandomInt(minVal, maxVal);

			retVal = new JBrain(*parent);
			if (!retVal->setValueByName(brainValueName, mutVal))
			{
				std::cout << "Failed to set value by name: " << brainValueName << std::endl;
				delete retVal;
				retVal = nullptr;
			}
		}

		return retVal;
	}

	JBrain_Snap* JBrainFactory::getUIntMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param)
	{
		int minVal, maxVal;
		getMinMaxIntFromConfig(minVal, maxVal, std::get<1>(param));
		JBrain_Snap* retVal = nullptr;

		// If there is room between the max and min, select a new random variable:
		if ((maxVal - minVal) >= MIN_INT_MUTATE_DIFF)
		{
			// Occasionally, this may not modify the value:
			int mutVal = getRandomInt(minVal, maxVal);
			
			// Make the new brain:
			retVal = new JBrain_Snap(*parent);
			if (!retVal->setValueByName(std::get<0>(param), static_cast<unsigned int>(mutVal)))
			{
				std::cout << "Failed to set UInt value by name: " << std::get<0>(param) << std::endl;
				delete retVal;
				retVal = nullptr;
			}
			else
				retVal->setValueByName("Name", getNextBrainName());
		}

		return retVal;
	}

	JBrain* JBrainFactory::getMutatedBrain_bool(JBrain* parent,
		const YAML::Node& config, const std::string& configValueName,
		const std::string& brainValueName)
	{
		std::string configString = getConfigAsString(config, configValueName);
		
		// Null means not modifiable:
		JBrain* retVal = nullptr;

		if (configString == "mutable")
		{
			retVal = new JBrain(*parent);

			// Mutable boolean, we can just tell the new child to swap the value:
			retVal->setValueByName(brainValueName, 1.0, true);
		}

		return retVal;
	}

	JBrain_Snap* JBrainFactory::getBoolMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param)
	{
		// If we can't mutate the brain, we give back null:
		JBrain_Snap* retVal = nullptr;

		std::string boolString = getValueAsString(std::get<1>(param), true);
		if (boolString == "mutable")
		{
			// Variable is changeable, create a new brain and tell it to swap that value:
			retVal = new JBrain_Snap(*parent);
			if (!retVal->setValueByName(std::get<0>(param), true, true))
			{
				std::cout << "Failed to set bool value by name " << std::get<0>(param) << std::endl;
			}
			else
				retVal->setValueByName("Name", getNextBrainName());
		}			

		return retVal;
	}

	std::vector<JBrain_Snap*> JBrainFactory::getFullMutatedPopulation(JBrain_Snap* parent)
	{
		// Vectors of parameters to try mutating on:
		std::vector<namePathTuple> doubleParams = getAllDoubleMutationParameters_snap();
		std::vector<namePathTuple> uIntParams = getAllUIntMutationParameters_snap();
		std::vector<namePathTuple> boolParams = getAllBoolMutationParameters_snap();
		std::vector<namePathTuple> stringListParams = getAllStringListMutationParameters_snap();
		
		// The full population we'll return:
		std::vector<JBrain_Snap*> retPop;
		JBrain_Snap* tempBrain;

		// Loop over each parameter set:
		for (auto npt : doubleParams)
		{
			tempBrain = getDoubleMutatedBrain_snap(parent, npt);
			if (tempBrain != nullptr)
				retPop.push_back(tempBrain);
		}
		
		for (auto npt : uIntParams)
		{
			tempBrain = getUIntMutatedBrain_snap(parent, npt);
			if (tempBrain != nullptr)
				retPop.push_back(tempBrain);
		}

		for (auto npt : boolParams)
		{
			tempBrain = getBoolMutatedBrain_snap(parent, npt);
			if (tempBrain != nullptr)
				retPop.push_back(tempBrain);
		}

		for (auto npt : stringListParams)
		{
			tempBrain = getStringMutatedBrain_snap(parent, npt);
			if (tempBrain != nullptr)
				retPop.push_back(tempBrain);
		}

		// Don't let the parent try again, rather recreate the same brain from the parent's
		// parameters and add it back to the population:
		tempBrain = new JBrain_Snap(*parent);
		delete parent;		
		tempBrain->setValueByName("Name", getNextBrainName());
		retPop.push_back(tempBrain);
		
		return retPop;
	}

	std::vector<JBrain*> JBrainFactory::getFullMutatedPopulation(JBrain* parent)
	{
		typedef std::vector<std::vector<std::string> > paramList;

		// This was being done as tuples of Yaml nodes and 2 strings, but it didn't
		// want to behave.  This is less elegant solution:
		static paramList floatMutations_dendrite{
			{ "MaxLength", "DendriteMaxLength" },
			{ "MinWeight", "DendriteMinWeight" },
			{ "MaxWeight", "DendriteMaxWeight" },
			{ "LowMoveAway", "DendriteLowMoveAway" },
			{ "HighMoveToward", "DendriteHighMoveToward" },
			{ "MoveAmount", "DendriteAwayTowardMoveAmount" },
			{ "LowWeightDecrease", "DendriteLowWeightDecrease" },
			{ "HighWeightIncrease", "DendriteHighWeightIncrease" },
			{ "WeightChangeAmount", "DendriteWeightChangeAmount" }
		};

		static std::vector<std::vector<std::string> > floatMutations_axon{
			{ "MaxLength", "AxonMaxLength" },
			{ "LowMoveAway", "AxonLowMoveAway" },
			{ "HighMoveToward", "AxonHighMoveToward" },
			{ "MoveAmount", "AxonAwayTowardMoveAmount" }
		};

		static std::vector<std::vector<std::string> > floatMutations_neuron{
			{ "FireThreshold", "NeuronFireThreshold" },
			{ "MinNearbyDistance", "NeuronDuplicateMinNearbyDistance" },
			{ "MaxNearbyDistance", "NeuronDuplicateMaxNearbyDistance" },
			{ "StartingDuplicateHealth", "NeuronDuplicateHealth" },
			{ "StartingDeathHealth", "NeuronDeathHealth" },
			{ "NeuronStartingHealth", "NeuronStartingHealth" },
			{ "LowHealthChange", "NeuronCGPOutputLowHealthChange" },
			{ "HighHealthChange", "NeuronCGPOutputHighHealthChange" },
			{ "HealthChangeAmount", "NeuronCGPOutputHealthChangeAmount" },
			{ "HealthThresholdMultiplier", "NeuronHealthThresholdMultiplier" },
			{ "MinFireValue", "NeuronMinFireValue" },
			{ "MaxFireValue", "NeuronMaxFireValue" },
			{ "NeuronSpaceDeteriorationParameter", "NeuronFireSpaceDeterioration" },
			{ "NeuronTimeDeteriorationParameter", "NeuronFireTimeDeterioration" },
			{ "FireThresholdIdleChange", "NeuronFireThresholdIdleChange" },
			{ "FireThresholdActiveChange", "NeuronFireThresholdActiveChange" }
		};

		static std::vector<std::vector<std::string> > floatMutations_brain{
			{ "Dimension", "BrainXSize" },
			{ "Dimension", "BrainYSize" },
			{ "Dimension", "BrainZSize" }
		};

		static std::vector<std::vector<std::string> > intMutations_dendrite{
			{ "MinCount", "DendriteMinCount" },
			{ "MaxCount", "DendriteMaxCount" }
		};

		static paramList intMutations_axon{
			{ "MinCount", "AxonMinCount" },
			{ "MaxCount", "AxonMaxCount" }
		};

		static paramList intMutations_neuron{
			{ "RefractoryPeriod", "NeuronRefractoryPeriod" },
			{ "NeuronFireLifetime", "NeuronFireLifetime" }
		};

		static paramList intMutations_brain{
			{ "ProcessingTimeStepsBetweenInputAndOutput", "BrainProcessingStepsBetweenInputAndOutput" },
			{ "OutputTimeStepsToAverageTogether", "BrainOutputsToAverageTogether" }
		};

		static paramList intMutations_equation{
			{ "UpdateProgramsFrequency", "UpdateProgramFrequency" }
		};

		static paramList boolMutations_neuron{
			{ "FireProbabilistic", "NeuronProbabilisticFire" },
			{ "NeuronDuplicatesNearby", "NeuronDuplicatesNearby" },
			{ "UseDynamicFireThresholds", "NeuronUseDynamicFireThresholds" }
		};

		static paramList boolMutations_sleep{
			{ "UsePreTrainSleep", "UsePreTrainsSleep" },
			{ "UsePostTrainSleep", "UsePostTrainSleep" }
		};

		static paramList boolMutations_brain{
			{ "OutputsOnOneSide", "BrainOutputsOnOneSide" },
			{ "InputsOnOneSide", "BrainInputsOnOneSide" },
			{ "OutputsIgnoreEnvironmentInputs", "BrainOutputsIgnoreEnvironmentInputs" },
			{ "UseSameValueForAllDimensions", "BrainUseSameDimensions" },
			{ "ResetBeforeProcessingInput", "ResetBeforeProcessingInput" },
			{ "InputsAge", "InputNeuronFiresAge" }
		};

		static paramList boolMutations_equation{};  // None so far.
		
		// The full next generation:
		std::vector<JBrain*> retVal;

		// Do all float mutations:
		for (const auto& mutVal : floatMutations_dendrite)
		{
			JBrain* temp = getMutatedBrain_float(parent, m_dendriteConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : floatMutations_axon)
		{
			JBrain* temp = getMutatedBrain_float(parent, m_axonConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}
		
		for (const auto& mutVal : floatMutations_neuron)
		{
			JBrain* temp = getMutatedBrain_float(parent, m_neuronConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : floatMutations_brain)
		{
			JBrain* temp = getMutatedBrain_float(parent, m_brainConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		// Do all integer mutations:
		for (const auto& mutVal : intMutations_dendrite)
		{
			JBrain* temp = getMutatedBrain_int(parent, m_dendriteConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : intMutations_axon)
		{
			JBrain* temp = getMutatedBrain_int(parent, m_axonConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : intMutations_neuron)
		{
			JBrain* temp = getMutatedBrain_int(parent, m_neuronConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}
		for (const auto& mutVal : intMutations_brain)
		{
			JBrain* temp = getMutatedBrain_int(parent, m_brainConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : intMutations_equation)
		{
			JBrain* temp = getMutatedBrain_int(parent, m_equationsConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		// Do all boolean mutations:
		for (const auto& mutVal : boolMutations_neuron)
		{
			JBrain* temp = getMutatedBrain_bool(parent, m_neuronConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : boolMutations_sleep)
		{
			JBrain* temp = getMutatedBrain_bool(parent, m_sleepConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : boolMutations_brain)
		{
			JBrain* temp = getMutatedBrain_bool(parent, m_brainConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		for (const auto& mutVal : boolMutations_equation)
		{
			JBrain* temp = getMutatedBrain_bool(parent, m_equationsConfig,
				mutVal[0], mutVal[1]);

			if (temp != nullptr)
				retVal.push_back(temp);
		}

		// If there is more than one JNeuron activation function, mutate on that:
		if (m_neuronActivationFunctions.size() > 1)
		{
			JBrain* temp = new JBrain(*parent);
			CGP::JNEURON_ACTIVATION_FUNCTION initial = temp->getJNeuronActivationFunction();
			CGP::JNEURON_ACTIVATION_FUNCTION mutVal = getRandomJNeuronActivationFunction();

			// Make sure we are actually changing the value:
			while (initial == mutVal)
				mutVal = getRandomJNeuronActivationFunction();

			temp->setValue(mutVal);

			retVal.push_back(temp);
		}

		// Mutate each CGP program:
		if (parent->m_CGPAxonUpdater != nullptr)
		{
			JBrain* temp = new JBrain(*parent);
			temp->m_CGPAxonUpdater->mutateSelf(CGP::MUTATION_STRATEGY::ACTIVE_GENE);
			retVal.push_back(temp);
		}

		if (parent->m_CGPDendriteUpdater != nullptr)
		{
			JBrain* temp = new JBrain(*parent);
			temp->m_CGPDendriteUpdater->mutateSelf(CGP::MUTATION_STRATEGY::ACTIVE_GENE);
			retVal.push_back(temp);
		}

		if (parent->m_CGPChemicalUpdater != nullptr)
		{
			JBrain* temp = new JBrain(*parent);
			temp->m_CGPChemicalUpdater->mutateSelf(CGP::MUTATION_STRATEGY::ACTIVE_GENE);
			retVal.push_back(temp);
		}

		if (parent->m_CGPNeuronUpdater != nullptr)
		{
			JBrain* temp = new JBrain(*parent);
			temp->m_CGPNeuronUpdater->mutateSelf(CGP::MUTATION_STRATEGY::ACTIVE_GENE);
			retVal.push_back(temp);
		}

		// Set all of the child brain's parent-names to the parent:
		for (auto& brain : retVal)
			brain->m_parentName = parent->m_name;

		// Add the parent back into the population:
		retVal.push_back(parent);

		// Reset the neurons of every brain (including the parent). We want
		// good parameters for learning, not a lucky starting configuration.
		// We also give the parent a new name since it is effectively a new
		// brain and a new test of its parameters.
		for (auto& brain : retVal)
		{
			brain->handleBrainSizeChange();
			brain->m_name = getNextBrainName();

			// Create either pre-defined or random neurons:
			if (m_staticNeuronsDefined)
				brain->setNeuronsFromStaticJson(m_staticNeuronsJson, false);
			else // Tell the brain to create enough random neurons:
				brain->addRandomStartingNeurons();

			// Output neurons, if defined, are already created. This will overwrite them:
			if (brain->getUseOutputNeurons() && m_staticOutputNeuronsDefined)
				brain->setNeuronsFromStaticJson(m_staticOutputNeuronsJson, true);
		}

		return retVal;
	}

} // End JBrain namespace