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

	JBrainFactory* JBrainFactory::getInstance()
	{
		// Not built to handle the potential of multiple threads:
		static JBrainFactory* instance = new JBrainFactory();
		return instance;
	}

	JBrainFactory::JBrainFactory() :
		m_dendriteConfig(YAML::Null),
		m_axonConfig(YAML::Null),
		m_neuronConfig(YAML::Null),
		m_sleepConfig(YAML::Null),
		m_brainConfig(YAML::Null),
		m_circuitsConfig(YAML::Null),
		m_equationsConfig(YAML::Null),
		m_initialized(false),
		m_currentBrainNumber(1),
		m_requiredFunctions(),
		m_mutableFunctions()
	{}

	JBrainFactory::~JBrainFactory() {}

	bool JBrainFactory::initialize(const std::string& yamlFilename)
	{
		// Load the full yaml:
		YAML::Node fullConfig = YAML::LoadFile(yamlFilename);

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
			std::cout << yamlFilename << " must include a 'Dendrite' section." << std::endl;
			goodConfig = false;
		}

		// Axon:
		if (fullConfig["Axon"])
			goodConfig = checkAxonConfig(fullConfig["Axon"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include an 'Axon' section" << std::endl;
			goodConfig = false;
		}

		// Neuron:
		if (fullConfig["Neuron"])
			goodConfig = checkNeuronConfig(fullConfig["Neuron"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include an 'Neuron' section" << std::endl;
			goodConfig = false;
		}

		// Sleep:
		if (fullConfig["Sleep"])
			goodConfig = checkSleepConfig(fullConfig["Sleep"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include a 'Sleep' section" << std::endl;
			goodConfig = false;
		}

		// Brain:
		if (fullConfig["Brain"])
			goodConfig = checkBrainConfig(fullConfig["Brain"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include a 'Brain' section" << std::endl;
			goodConfig = false;
		}

		// Circuit:
		if (fullConfig["Circuit"])
			goodConfig = checkCircuitConfig(fullConfig["Circuit"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include a 'Circuit' section" << std::endl;
			goodConfig = false;
		}

		// Equations:
		if (fullConfig["Equation"])
			goodConfig = checkEquationsConfig(fullConfig["Equation"]) && goodConfig;
		else
		{
			std::cout << yamlFilename << " must include an 'Equation' section" << std::endl;
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

	bool JBrainFactory::checkCircuitConfig(const YAML::Node& config)
	{
		m_circuitsConfig = config;
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

	JBrain* JBrainFactory::getRandomBrain()
	{
		if (!m_initialized)
		{
			std::cout << "Asking for a brain from uninitiaziled factory." << std::endl;
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
			getConfigAsMutableBool(m_brainConfig, "InputsOnOneSide"), // brainInputsOnOneSide
			getConfigAsMutableBool(m_brainConfig, "OutputsOnOneSide"), // brainOutputsOnOneSide
			getFloatFromConfigRange(m_circuitsConfig, "MinMinDimension", "MaxMinDimension"),  // circuitMinDimensions
			getFloatFromConfigRange(m_circuitsConfig, "MinMaxDimension", "MaxMaxDimension"),  // circuitMaxDimensions
			getConfigAsMutableBool(m_circuitsConfig, "UseSameValueForAllDimensions"),  //circuitUseSameDimensions
			static_cast<unsigned int>(getIntFromConfigRange(m_circuitsConfig, "MinMinCircuitCount", "MaxMinCircuitCount")), // circuitMinCircuitCount
			static_cast<unsigned int>(getIntFromConfigRange(m_circuitsConfig, "MinMaxCircuitCount", "MaxMaxCircuitCount")), // circuitMaxCircuitCount
			getFloatFromConfigRange(m_circuitsConfig, "MinProbabilityIndividualCircuitIsPassedToChild", "MaxProbabilityIndividualCircuitIsPassedToChild"), //circuitProbabilityCircuitPassedToChild
			// circuitProbabilityFireChangeWhenOtherNeuronFires
			getFloatFromConfigRange(m_circuitsConfig, "MinFireProbabilityChangeDueToOtherNeuronsInTheSameCircuitFiring",
				                    "MaxFireProbabilityChangeDueToOtherNeuronsInTheSameCircuitFiring"),			
			getFloatFromConfigRange(m_circuitsConfig, "MinNeuronDuplicatesInSameCircuit", "MaxNeuronDuplicatesInSameCircuit"), // circuitProbabilityNeuronDuplicateInCircuit
			getFloatFromConfigRange(m_circuitsConfig, "MinNeuronHealthChangeFromDeath", "MaxNeuronHealthChangeFromDeath"), // circuitNeuronHealthChangeFromNeuronDeath
			getFloatFromConfigRange(m_circuitsConfig, "MinNeuronHealthChangeFromDuplication", "MaxNeuronHealthChangeFromDuplication"), // circuitNeuronHealthChangeFromNeuronDuplication
			getConfigAsMutableBool(m_circuitsConfig, "CircuitsCanOverlap"), // circuitsCanOverlap
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
			retVal->setValueByName(brainValueName, mutVal);
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
			{ "ProcessingTimeStepsBetweenInputAndOutput", "BrainProcessingStepsBetweenInputAndOutput" }
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