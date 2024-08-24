#pragma once
#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <sys/wait.h>
#endif

#include <string>
#include <unordered_map>

#include "yaml-cpp/yaml.h"

namespace Experiment
{
	/**************
	* A MultiExperiment will take a YAML file that describes multiple
	* Experiments and run each Experiment as a separate process.
	* Multithreading can't be used due to the Python API's need to only
	* have a single line of code running in any given process at any
	* given time.
	*************/
	class MultiExperiment
	{
	public:
		MultiExperiment();

		void runDirectoryOfExperiments(std::string directory, unsigned int maxConcurrentChildren);

		// Separate the yaml file into multiple files. Returns the directory
		// where those files are stored.
		std::string setupExperiments(std::string yamlFilename);

		std::vector<YAML::Node> separateYamlConfigs(std::string yamlFilename);
		std::vector<YAML::Node> separateYamlConfigs(YAML::Node inNode);

		// Write out multiple yamls with numbered file names:
		void saveMultipleYaml(const std::vector<YAML::Node>& manyYaml, const std::string& origFilename);

		// Check that the values that all of the min-max rules aren't broken:
		bool getIsNumericallyValid(const YAML::Node& testYaml);
	
		void recursiveGetNamedValuePairs(const YAML::Node& fullYaml,
			std::unordered_map<std::string, double>& namedVals,
			std::vector<std::string>& currentDepth);

		std::vector<YAML::Node> getNumericallyValidYaml(const std::vector<YAML::Node>& manyYaml);

	protected:
		void recursiveSeparate(const YAML::Node& fullFile,
			std::vector<YAML::Node>& outputCopies,
			std::vector<std::string>& currentDepth);

		YAML::Node getNodeAtDepth(const YAML::Node& top, const std::vector<std::string>& depth);
		
		void setValue(YAML::Node& top, const std::vector<std::string>& depth, const std::string& value);
		void setValue(YAML::Node& top, const std::vector<std::string>& depth, const YAML::Node& value);

		// Duplicate the currentList X times.
		void addCopies(std::vector<YAML::Node>& currentList, unsigned int copies);
				
		// DEBUGGING
		void copyExample(std::string yamlFilename);
		void stringVectorOut(const std::vector<std::string>& in);

	private:

		std::string getCurrentTimeString();
		
		// This will track the min/max pairs that can't be violated
		// in a valid YAML
		static std::unordered_map<std::string, std::string> m_yamlMinMaxPairs;

		// Multiprocessing code below this comment.
#ifdef __linux__
		pid_t startSingleChild(std::vector<std::string>& yamlVector);
		void doWork(const std::vector<std::string>& yamlVector);
#elif _WIN32
		bool startSingleChild(std::vector<STARTUPINFOA>& startInfos,
			std::vector<PROCESS_INFORMATION>& processInfos,
			std::vector<std::string>& workToDo);
		bool checkRunningAndRemove(std::vector<STARTUPINFOA>& startInfos,
			std::vector<PROCESS_INFORMATION>& processInfos,
			unsigned int processToCheck);
#endif

	};
}

