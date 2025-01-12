#pragma once

#include "JBrain.h"
#include "Enums.h"
#include <string>
#include <vector>
#include <functional>
#include "yaml-cpp/yaml.h"
#include "JsonLib/json.hpp"
using json = nlohmann::json;
namespace JBrain
{
	// Typedef for our various mutate types. Each is the name the brain
	// expects followed by the path to reach it in the yaml file:
	typedef std::tuple<std::string, std::vector<std::string> > namePathTuple;

	class JBrainFactory
	{
	private:
		// How far apart floating point variables need to be to warrant
		// using them in mutation:
		static const float MIN_FLOAT_MUTATE_DIFF;
		static const int MIN_INT_MUTATE_DIFF = 1;
		static const double	MIN_DOUBLE_MUTATE_DIFF;
		static const int MIN_STRING_LIST_LENGTH = 2;

		// Private constructor for singleton class:
		JBrainFactory();

		virtual const std::string classname() { return "JBrainFactory"; }

		// These functions will each check a section of the configuration yaml and ensure
		// that the provided options there satisfy all required conditions. They are only
		// used by the growth paradigm.
		bool checkDendriteConfig(const YAML::Node& config);
		bool checkAxonConfig(const YAML::Node& config);
		bool checkNeuronConfig(const YAML::Node& config);
		bool checkSleepConfig(const YAML::Node& config);
		bool checkBrainConfig(const YAML::Node& config);
		bool checkEquationsConfig(const YAML::Node& config);

		void buildDendriteCGPLists();
		void buildAxonCGPLists();
		void buildNeuronCGPLists();

		// Initialize this class based on the brain paradigm used:
		bool initialize_growth(const YAML::Node& fullConfig);
		bool initialize_snap(const YAML::Node& fullConfig);

		// Variables related to using static training data rather than interaction with an environment:
		bool m_csvTrainingDataProvided;
		std::vector<std::vector<double> > m_csvObservations;
		std::vector<std::vector<double> > m_csvActions;
		std::vector<double> m_observationMinimums;
		std::vector<double> m_observationMaximums;

		// Make sure the values are far enough apart:
		bool getDoubleConfigDifferentValues(const std::vector<std::string> path);
		bool getIntConfigDifferentValues(const std::vector<std::string> path);

		// Get min/max values from a snap-config list:
		bool getMinMaxDoubleFromConfig(double& outMin, double& outMax, const std::vector<std::string> path);
		bool getMinMaxIntFromConfig(int& outMin, int& outMax, const std::vector<std::string> path);
		
		// Special cases:
		CGP::DYNAMIC_PROBABILITY getRandomDynamicProbabilityApplication();
		CGP::HDC_LEARN_MODE getRandomHDCLearnMode();

		// Get a random string from a list of strings:
		std::string getConfigStringFromListOfStrings(const std::vector<std::string>& fullPath);

		// Get learning data from a CSV rather than interacting with an environment:
		bool readTrainingCSV(std::string& filename, unsigned int& obsSize, unsigned int& actSize);

	protected:
		// Snap information:
		YAML::Node m_fullConfig; // Keep the top config, store full paths to what we need.		
		
		// Growth information:
		YAML::Node m_dendriteConfig;
		YAML::Node m_axonConfig;
		YAML::Node m_neuronConfig;
		YAML::Node m_sleepConfig;
		YAML::Node m_brainConfig;
		YAML::Node m_equationsConfig;

		bool m_initialized;
		unsigned int m_currentBrainNumber; // For providing unique brain IDs.

		// Information on static neuron definitions:
		bool m_staticNeuronsDefined;
		bool m_staticOutputNeuronsDefined;
		json m_staticNeuronsJson;
		json m_staticOutputNeuronsJson;

		// Functions available to the JBrainCGPIndividuals:
		std::vector<std::string> m_requiredFunctions;  // Those set to "true"
		std::vector<std::string> m_mutableFunctions;  // Those set to mutable
		void buildFunctionLists();  // Fill in the above vectors from m_equationsConfig

		// Activation functions available to JNeurons:
		std::vector<CGP::JNEURON_ACTIVATION_FUNCTION> m_neuronActivationFunctions;

		// The input and output lists to the CGP programs:
		std::vector<CGP::CGP_INPUT> m_dendriteInputs;
		std::vector<CGP::CGP_OUTPUT> m_dendriteOutputs;
		std::vector<CGP::CGP_INPUT> m_outputDendriteInputs;
		std::vector<CGP::CGP_OUTPUT> m_outputDendriteOutputs;
		std::vector<CGP::CGP_INPUT> m_axonInputs;
		std::vector<CGP::CGP_OUTPUT> m_axonOutputs;
		std::vector<CGP::CGP_INPUT> m_neuronInputs;
		std::vector<CGP::CGP_OUTPUT> m_neuronOutputs;

		// Stored observation processor
		ObservationProcessor* m_observationProcessor;

		std::string getNextBrainName();
		ObservationProcessor* getObservationProcessor();
		
		// Get variables from our configuration YAML nodes:
		float getConfigAsFloat(const YAML::Node& config, const std::string& name);
		std::string getConfigAsString(const YAML::Node& config, const std::string& name, bool convertToLowercase = true);
		int getConfigAsInt(const YAML::Node& config, const std::string& name);
		// Get a boolean value from the configuration and randomize the value if "mutable" is found:
		bool getConfigAsMutableBool(const YAML::Node& config, const std::string& name);
		// Given two config values, choose a random number between them. These functions just make it
		// syntactically easier to get the random values we need:
		float getFloatFromConfigRange(const YAML::Node& config, const std::string& minName, const std::string& maxName);
		int getIntFromConfigRange(const YAML::Node& config, const std::string& minName, const std::string& maxName);
		

		// Fill a vector with all of the mutatable parameters:
		std::vector<namePathTuple> getAllDoubleMutationParameters_snap();
		std::vector<namePathTuple> getAllUIntMutationParameters_snap();
		std::vector<namePathTuple> getAllBoolMutationParameters_snap();
		std::vector<namePathTuple> getAllStringListMutationParameters_snap();

		// Mutate a brain based on a namePathTuple:
		JBrain_Snap* getDoubleMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param);
		JBrain_Snap* getUIntMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param);
		JBrain_Snap* getBoolMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param);
		JBrain_Snap* getStringMutatedBrain_snap(const JBrain_Snap* parent, const namePathTuple& param);

		// Get config from Snap-brain YAML, which uses lists:
		double getRandomListConfigAsDouble(const std::vector<std::string>& fullPath);
		int getRandomListConfigAsInt(const std::vector<std::string>& fullPath);
		bool getRandomConfigAsBool(const std::vector<std::string>& fullPath);
		std::vector<std::string> getListOfStrings(std::vector<std::string> fullPath);
		std::string getValueAsString(const std::vector<std::string>& fullPath, bool convertToLowercase=true);
		int getValueAsInt(const std::vector<std::string>& fullPath);

		// Given the available required and mutable functions, generate a valid
		// random list of functions and their corresponding names:
		void getRandomCGPFunctionLists(std::vector<std::string>& functionNames,
			std::vector<std::function<double(double, double, double)> >& functions);

		JBrain* getMutatedBrain_float(JBrain* parent, const YAML::Node& config,
			const std::string& configValueName, const std::string& brainValueName);

		JBrain* getMutatedBrain_int(JBrain* parent, const YAML::Node& config,
			const std::string& configValueName, const std::string& brainValueName);

		JBrain* getMutatedBrain_bool(JBrain* parent, const YAML::Node& config,
			const std::string& configValueName, const std::string& brainValueName);

		CGP::JNEURON_ACTIVATION_FUNCTION getRandomJNeuronActivationFunction();

	public:
		void setNextBrainNumber(unsigned int currBrainNum) { m_currentBrainNumber = currBrainNum; }

		static JBrainFactory* getInstance();
		float getRandomFloat(const float& min, const float& max);
		double getRandomDouble(const double& min, const double& max);
		int getRandomInt(const int& min, const int& max);
		bool getRandomBool();
		bool getHDCModeSet();
		
		// Trusting the requesters to not screw with our data:
		bool getCSVTrainingDataProvided() { return m_csvTrainingDataProvided; }
		std::vector<std::vector<double> >* getCSVObservations() { return &m_csvObservations; }
		std::vector<std::vector<double> >* getCSVActions() { return &m_csvActions; }

		~JBrainFactory();

		bool initialize(const std::string& yamlFilename);
		YAML::Node getExperimentConfig(); // Get the experiment configuration

		JBrain* getRandomBrain();
		JBrain_Snap* getRandomSnapBrain();

		// Make one new brain for each mutatable parameter:
		std::vector<JBrain*> getFullMutatedPopulation(JBrain* parent);
		std::vector<JBrain_Snap*> getFullMutatedPopulation(JBrain_Snap* parent);
	};

} // End JBrain Namespace