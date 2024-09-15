#pragma once

#include "JBrain.h"
#include "Enums.h"
#include <string>
#include <vector>
#include <functional>
#include "yaml-cpp/yaml.h"

namespace JBrain
{
	class JBrainFactory
	{
	private:
		// How far apart floating point variables need to be to warrant
		// using them in mutation:
		static const float MIN_FLOAT_MUTATE_DIFF;
		static const int MIN_INT_MUTATE_DIFF = 1;

		// Private constructor for singleton class:
		JBrainFactory();

		virtual const std::string classname() { return "JBrainFactory"; }

		// These functions will each check a section of the configuration yaml and ensure
		// that the provided options there satisfy all required conditions:
		bool checkDendriteConfig(const YAML::Node& config);
		bool checkAxonConfig(const YAML::Node& config);
		bool checkNeuronConfig(const YAML::Node& config);
		bool checkSleepConfig(const YAML::Node& config);
		bool checkBrainConfig(const YAML::Node& config);
		bool checkCircuitConfig(const YAML::Node& config);
		bool checkEquationsConfig(const YAML::Node& config);

		void buildDendriteCGPLists();
		void buildAxonCGPLists();
		void buildNeuronCGPLists();

	protected:
		YAML::Node m_dendriteConfig;
		YAML::Node m_axonConfig;
		YAML::Node m_neuronConfig;
		YAML::Node m_sleepConfig;
		YAML::Node m_brainConfig;
		YAML::Node m_circuitsConfig;
		YAML::Node m_equationsConfig;

		bool m_initialized;
		unsigned int m_currentBrainNumber; // For providing unique brain IDs.

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

		std::string getNextBrainName();
		
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
		static JBrainFactory* getInstance();
		float getRandomFloat(const float& min, const float& max);
		int getRandomInt(const int& min, const int& max);
		bool getRandomBool();

		~JBrainFactory();

		bool initialize(const std::string& yamlFilename);

		JBrain* getRandomBrain();

		// Make one new brain for each mutatable parameter:
		std::vector<JBrain*> getFullMutatedPopulation(JBrain* parent);
	};

} // End JBrain Namespace