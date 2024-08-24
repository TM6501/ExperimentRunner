#pragma once
# include <string>

#include "yaml-cpp/yaml.h"

namespace Experiment
{
	// This interface defines what interfaces persistent worlds must
	// provide. Persistent worlds are those for which there is a single
	// universe in which all agents exist throughout an experiment.

	class IPersistentWorld
	{
	public:
		virtual ~IPersistentWorld();

		// Load all of what we need from a Yaml file. Return
		// True if we are successful and found all data we needed:
		virtual bool loadConfigurationFromYaml(YAML::Node node) = 0;
	};
}