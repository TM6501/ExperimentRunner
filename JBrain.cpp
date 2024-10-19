#include "pch.h"

#include "JBrain.h"
#include "JBrainCGPIndividual.h"
#include "CGPFunctions.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <sstream>
#include "yaml-cpp/yaml.h"

namespace JBrain
{
	// Construct used by JBrainFactory requires every parameter.
	// They must be provided in the same order as they are declared
	// in the header file to prevent Visual Studio from having a conniption.
	JBrain::JBrain(const std::string& name,
		const std::string& parentName,
		const unsigned int& observationSize,
		const unsigned int& actionSize,
		const float& dendriteMaxLength,
		const unsigned int& dendriteMinCount,
		const unsigned int& dendriteMaxCount,
		const float& dendriteMinWeight,
		const float& dendriteMaxWeight,
		const float& dendriteLowMoveAway,
		const float& dendriteHighMoveToward,
		const float& dendriteAwayTowardMoveAmount,
		const float& dendriteLowWeightDecrease,
		const float& dendriteHighWeightIncrease,
		const float& dendriteWeightChangeAmount,
		const float& axonMaxLength,
		const unsigned int& axonMinCount,
		const unsigned int& axonMaxCount,
		const float& axonLowMoveAway,
		const float& axonHighMoveToward,
		const float& axonAwayTowardMoveAmount,
		const bool& neuronProbabilisticFire,
		const float& neuronFireThreshold,
		const float& neuronMinFireValue,
		const float& neuronMaxFireValue,
		const bool& neuronUseDynamicFireThresholds,
		const float& neuronFireThresholdIdleChange,
		const float& neuronFireThresholdActiveChange,
		const unsigned int& neuronRefractoryPeriod,
		const bool& neuronDuplicateNearby,
		const float& neuronMinNearbyDistance,
		const float& neuronMaxNearbyDistance,
		const unsigned int& minStartingNeurons,
		const unsigned int& maxStartingNeurons,
		const unsigned int& maxNeurons,
		const bool& useOutputNeurons,
		const float& neuronStartingHealth,
		const float& neuronCGPOutputLowHealthChange,
		const float& neuronCGPOutputHighHealthChange,
		const float& neuronCGPOutputHealthChangeAmount,
		const float& neuronDeathHealth,
		const float& neuronDuplicateHealth,
		const float& neuronDeathDuplicateHealthThresholdMultiplier,
		const float& neuronDuplicationHealthChange,
		const bool& neuronDuplicationHealthReset,
		const CGP::JNEURON_ACTIVATION_FUNCTION& neuronActivationFunction,
		const float& neuronFireSpaceDeterioration,
		const float& neuronFireTimeDeterioration,
		const unsigned int& neuronFireLifetime,
		const bool& inputNeuronFiresAge,
		const bool& usePreTrainSleep,
		const bool& usePostTrainSleep,
		const float& brainXSize,
		const float& brainYSize,
		const float& brainZSize,
		const bool& brainUseSameDimensions,
		const bool& brainResetBeforeProcessingInput,
		const unsigned int& brainProcessingStepsBetweenInputAndOutput,
		const unsigned int& brainOutputsToAverageTogether,
		const bool& brainInputsOnOneSide,
		const bool& brainOutputsOnOneSide,
		const bool& brainOutputsIgnoreEnvironmentInputs,
		const float& minP, const float& maxP,
		const float& minConstraint, const float& maxConstraint,
		const unsigned int& maxNeuronAge,
		const std::vector<CGP::CGP_INPUT>& dendriteInputs,
		const std::vector<CGP::CGP_OUTPUT>& dendriteOutputs,
		const unsigned int& dendriteProgramNodes,
		const std::vector<CGP::CGP_INPUT>& outputDendriteInputs,
		const std::vector<CGP::CGP_OUTPUT>& outputDendriteOutputs,
		const unsigned int& outputDendriteProgramNodes,
		const std::vector<CGP::CGP_INPUT>& axonInputs,
		const std::vector<CGP::CGP_OUTPUT>& axonOutputs,
		const unsigned int& axonProgramNodes,
		const std::vector<CGP::CGP_INPUT>& neuronInputs,
		const std::vector<CGP::CGP_OUTPUT>& neuronOutputs,
		const unsigned int& neuronProgramNodes,
		const CGP::UPDATE_EVENT& updateEvent,
		const unsigned int& updateFrequency,
		const std::vector<std::string>& functionStringList,
		const std::vector<std::function<double(double, double, double)> >& functionList,
		const bool& needInitializeUpdatersAndConnections):
	m_name(name),
		m_parentName(parentName),
		m_observationSize(observationSize),
		m_actionSize(actionSize),
		m_dendriteMaxLength(dendriteMaxLength),
		m_dendriteMinCount(dendriteMinCount),
		m_dendriteMaxCount(dendriteMaxCount),
		m_dendriteMinWeight(dendriteMinWeight),
		m_dendriteMaxWeight(dendriteMaxWeight),
		m_dendriteLowMoveAway(dendriteLowMoveAway),
		m_dendriteHighMoveToward(dendriteHighMoveToward),
		m_dendriteAwayTowardMoveAmount(dendriteAwayTowardMoveAmount),
		m_dendriteLowWeightDecrease(dendriteLowWeightDecrease),
		m_dendriteHighWeightIncrease(dendriteHighWeightIncrease),
		m_dendriteWeightChangeAmount(dendriteWeightChangeAmount),
		m_axonMaxLength(axonMaxLength),
		m_axonMinCount(axonMinCount),
		m_axonMaxCount(axonMaxCount),
		m_axonLowMoveAway(axonLowMoveAway),
		m_axonHighMoveToward(axonHighMoveToward),
		m_axonAwayTowardMoveAmount(axonAwayTowardMoveAmount),
		m_neuronProbabilisticFire(neuronProbabilisticFire),
		m_neuronFireThreshold(neuronFireThreshold),
		m_neuronMinFireValue(neuronMinFireValue),
		m_neuronMaxFireValue(neuronMaxFireValue),
		m_neuronUseDynamicFireThresholds(neuronUseDynamicFireThresholds),
		m_neuronFireThresholdIdleChange(neuronFireThresholdIdleChange),
		m_neuronFireThresholdActiveChange(neuronFireThresholdActiveChange),
		m_neuronRefractoryPeriod(neuronRefractoryPeriod),
		m_neuronDuplicateNearby(neuronDuplicateNearby),
		m_neuronMinNearbyDistance(neuronMinNearbyDistance),
		m_neuronMaxNearbyDistance(neuronMaxNearbyDistance),
		m_minStartingNeurons(minStartingNeurons),
		m_maxStartingNeurons(maxStartingNeurons),
		m_maxNeurons(maxNeurons),
		m_useOutputNeurons(useOutputNeurons),
		m_neuronStartingHealth(neuronStartingHealth),
		m_neuronCGPOutputLowHealthChange(neuronCGPOutputLowHealthChange),
		m_neuronCGPOutputHighHealthChange(neuronCGPOutputHighHealthChange),
		m_neuronCGPOutputHealthChangeAmount(neuronCGPOutputHealthChangeAmount),
		m_neuronDeathHealth(neuronDeathHealth),
		m_neuronDeathHealth_Original(neuronDeathHealth),
		m_neuronDuplicateHealth(neuronDuplicateHealth),
		m_neuronDuplicateHealth_Original(neuronDuplicateHealth),
		m_neuronDuplicationHealthChange(neuronDuplicationHealthChange),
		m_neuronDuplicationHealthReset(neuronDuplicationHealthReset),
		m_jNeuronActivationFunction(neuronActivationFunction),
		m_neuronDeathDuplicateHealthThresholdMultiplier(neuronDeathDuplicateHealthThresholdMultiplier),
		m_neuronFireSpaceDeterioration(neuronFireSpaceDeterioration),
		m_neuronFireTimeDeterioration(neuronFireTimeDeterioration),
		m_neuronFireLifetime(neuronFireLifetime),
		m_inputNeuronFiresAge(inputNeuronFiresAge),
		m_currNeuronNumber(0),
		m_usePreTrainSleep(usePreTrainSleep),
		m_usePostTrainSleep(usePostTrainSleep),
		m_brainXSize(brainXSize),
		m_brainYSize(brainYSize),
		m_brainZSize(brainZSize),
		m_brainUseSameDimensions(brainUseSameDimensions),
		m_brainResetBeforeProcessingInput(brainResetBeforeProcessingInput),
		m_brainProcessingStepsBetweenInputAndOutput(brainProcessingStepsBetweenInputAndOutput),
		m_brainOutputsToAverageTogether(brainOutputsToAverageTogether),
		m_brainInputsOnOneSide(brainInputsOnOneSide),
		m_brainOutputsOnOneSide(brainOutputsOnOneSide),
		m_brainOutputsIgnoreEnvironmentInputs(brainOutputsIgnoreEnvironmentInputs),
		m_minP(minP), m_maxP(maxP),
		m_minConstraint(minConstraint), m_maxConstraint(maxConstraint),
		m_maxNeuronAge(maxNeuronAge),
		m_averageNeuronFirePercentage(0.0),
		m_sageMatchPercent(1.0),
		m_dendriteInputs(dendriteInputs),
		m_dendriteOutputs(dendriteOutputs),
		m_outputDendriteInputs(outputDendriteInputs),
		m_outputDendriteOutputs(outputDendriteOutputs),
		m_axonInputs(axonInputs),
		m_axonOutputs(axonOutputs),
		m_neuronInputs(neuronInputs),
		m_neuronOutputs(neuronOutputs),
		m_dendriteProgramNodes(dendriteProgramNodes),
		m_outputDendriteProgramNodes(outputDendriteProgramNodes),
		m_axonProgramNodes(axonProgramNodes),
		m_neuronProgramNodes(neuronProgramNodes),
	    m_CGPDendriteUpdater(nullptr),
		m_CGPOutputDendriteUpdater(nullptr),
		m_CGPAxonUpdater(nullptr),
		m_CGPNeuronUpdater(nullptr),
		m_CGPChemicalUpdater(nullptr),
		m_updateEvent(updateEvent),
		m_updateFrequency(updateFrequency),
		m_functionStringList(functionStringList),
		m_functionList(functionList),
		m_inputProcessingsSinceLastUpdate(0),
		m_totalTrialInputsProcessed(0),
		m_totalTrialSageChoiceMatches(0),
		m_outputCSV(nullptr)
	{
		// Same size required?
		if (m_brainUseSameDimensions)
		{
			if ((m_brainXSize != m_brainYSize) || (m_brainYSize != m_brainZSize))
			{
				/*
				std::cout << "Brain \"" << m_name << "\" requires same dimensions "
					<< " but was provided differing dimensions. X:"
					<< m_brainXSize << " Y:" << m_brainYSize << " Z:"
					<< m_brainZSize;
				*/
				float newBrainSize = std::max(m_brainXSize, std::max(m_brainYSize, m_brainZSize));
				
				// std::cout << " Setting all size values to " << newBrainSize << std::endl;

				m_brainXSize = m_brainYSize = m_brainZSize = newBrainSize;
			}			
		}

		// To ensure that all brains created with the same inputs and outputs
		// use the same ordering, sort them here:
		std::sort(m_dendriteInputs.begin(), m_dendriteInputs.end());
		std::sort(m_outputDendriteInputs.begin(), m_outputDendriteInputs.end());
		std::sort(m_axonInputs.begin(), m_axonInputs.end());
		std::sort(m_neuronInputs.begin(), m_neuronInputs.end());
		std::sort(m_dendriteOutputs.begin(), m_dendriteOutputs.end());
		std::sort(m_outputDendriteOutputs.begin(), m_outputDendriteOutputs.end());
		std::sort(m_axonOutputs.begin(), m_axonOutputs.end());
		std::sort(m_neuronOutputs.begin(), m_neuronOutputs.end());

		// If we're being read in from a file, the updaters and in/outs
		// will be provided after construction:
		if (needInitializeUpdatersAndConnections)
		{
			createAllSeparateUpdaters();
			createInputsAndOutputs();
		}
	}

	JBrain::JBrain(const JBrain& other)
		: JBrain(other.m_name, other.m_parentName, other.m_observationSize,
			other.m_actionSize, other.m_dendriteMaxLength,
			other.m_dendriteMinCount, other.m_dendriteMaxCount,
			other.m_dendriteMinWeight, other.m_dendriteMaxWeight,
			other.m_dendriteLowMoveAway, other.m_dendriteHighMoveToward,
			other.m_dendriteAwayTowardMoveAmount, other.m_dendriteLowWeightDecrease,
			other.m_dendriteHighWeightIncrease, other.m_dendriteWeightChangeAmount,
			other.m_axonMaxLength, other.m_axonMinCount,
			other.m_axonMaxCount, other.m_axonLowMoveAway,
			other.m_axonHighMoveToward, other.m_axonAwayTowardMoveAmount,
			other.m_neuronProbabilisticFire,
			other.m_neuronFireThreshold, other.m_neuronMinFireValue,
			other.m_neuronMaxFireValue, other.m_neuronUseDynamicFireThresholds,
			other.m_neuronFireThresholdIdleChange, other.m_neuronFireThresholdActiveChange,
			other.m_neuronRefractoryPeriod, other.m_neuronDuplicateNearby,
			other.m_neuronMinNearbyDistance, other.m_neuronMaxNearbyDistance,
			other.m_minStartingNeurons, other.m_maxStartingNeurons, other.m_maxNeurons,
			other.m_useOutputNeurons, other.m_neuronStartingHealth,
			other.m_neuronCGPOutputLowHealthChange,
			other.m_neuronCGPOutputHighHealthChange,
			other.m_neuronCGPOutputHealthChangeAmount,
			other.m_neuronDeathHealth_Original, other.m_neuronDuplicateHealth_Original,
			other.m_neuronDeathDuplicateHealthThresholdMultiplier,
			other.m_neuronDuplicationHealthChange, other.m_neuronDuplicationHealthReset,
			other.m_jNeuronActivationFunction, other.m_neuronFireSpaceDeterioration,
			other.m_neuronFireTimeDeterioration, other.m_neuronFireLifetime,
			other.m_inputNeuronFiresAge, other.m_usePreTrainSleep,
			other.m_usePostTrainSleep, other.m_brainXSize, other.m_brainYSize,
			other.m_brainZSize, other.m_brainUseSameDimensions,
			other.m_brainResetBeforeProcessingInput,
			other.m_brainProcessingStepsBetweenInputAndOutput,
			other.m_brainOutputsToAverageTogether,
			other.m_brainInputsOnOneSide, other.m_brainOutputsOnOneSide,
			other.m_brainOutputsIgnoreEnvironmentInputs, other.m_minP, other.m_maxP,
			other.m_minConstraint, other.m_maxConstraint, other.m_maxNeuronAge,
			other.m_dendriteInputs, other.m_dendriteOutputs,
			other.m_dendriteProgramNodes, other.m_outputDendriteInputs,
			other.m_outputDendriteOutputs, other.m_outputDendriteProgramNodes,
			other.m_axonInputs, other.m_axonOutputs, other.m_axonProgramNodes,
			other.m_neuronInputs, other.m_neuronOutputs, other.m_neuronProgramNodes,
			other.m_updateEvent, other.m_updateFrequency, other.m_functionStringList,
			other.m_functionList,
			false) // Need to create CGP & inputs/outputs.
	{
		// Create all CGP Updaters:
		if (other.m_CGPDendriteUpdater != nullptr)
			m_CGPDendriteUpdater = new CGP::JBrainCGPIndividual(*other.m_CGPDendriteUpdater);

		if (other.m_CGPOutputDendriteUpdater != nullptr)
			m_CGPOutputDendriteUpdater = new CGP::JBrainCGPIndividual(*other.m_CGPOutputDendriteUpdater);

		if (other.m_CGPAxonUpdater != nullptr)
			m_CGPAxonUpdater = new CGP::JBrainCGPIndividual(*other.m_CGPAxonUpdater);

		if (other.m_CGPNeuronUpdater != nullptr)
			m_CGPNeuronUpdater = new CGP::JBrainCGPIndividual(*other.m_CGPNeuronUpdater);

		if (other.m_CGPChemicalUpdater != nullptr)
			m_CGPChemicalUpdater = new CGP::JBrainCGPIndividual(*other.m_CGPChemicalUpdater);

		m_neurons = other.m_neurons;
		m_outputNeurons = other.m_outputNeurons;
		m_currNeuronNumber = other.m_currNeuronNumber;
		m_inputAxons = other.m_inputAxons;
		m_outputDendrites = other.m_outputDendrites;
	}

	JBrain::~JBrain()
	{
		if (m_CGPDendriteUpdater != nullptr)
			delete m_CGPDendriteUpdater;
		
		if (m_CGPOutputDendriteUpdater != nullptr)
			delete m_CGPOutputDendriteUpdater;

		if (m_CGPAxonUpdater != nullptr)
			delete m_CGPAxonUpdater;
		
		if (m_CGPNeuronUpdater != nullptr)
			delete m_CGPNeuronUpdater;

		if (m_CGPChemicalUpdater != nullptr)
			delete m_CGPChemicalUpdater;

		closeCSVOutputFile();
	}

	void JBrain::createDendriteUpdater()
	{
		m_CGPDendriteUpdater = new CGP::JBrainCGPIndividual(
			static_cast<unsigned int>(m_dendriteInputs.size()), // Number of inputs
			static_cast<unsigned int>(m_dendriteOutputs.size()), // Number of outputs
			1, // Number of rows
			m_dendriteProgramNodes, // Number of columns
			m_dendriteProgramNodes, // Columns back to search
			m_minP, m_maxP, // P-values for CGP functions
			m_functionStringList, // Available functions as strings
			m_functionList, // Available functions as callable (dbl, dbl, dbl) functions
			m_minConstraint, m_maxConstraint, true  // Node output constraints
		);
		m_CGPDendriteUpdater->randomize();
	}

	void JBrain::createOutputDendriteUpdater()
	{
		m_CGPOutputDendriteUpdater = new CGP::JBrainCGPIndividual(
			static_cast<unsigned int>(m_outputDendriteInputs.size()), // Number of inputs
			static_cast<unsigned int>(m_outputDendriteOutputs.size()), // Number of outputs
			1, // Number of rows
			m_outputDendriteProgramNodes, // Number of columns
			m_outputDendriteProgramNodes, // Columns back to search			
			m_minP, m_maxP, // P-values for CGP functions
			m_functionStringList, // Available functions as strings
			m_functionList, // Available functions as callable (dbl, dbl, dbl) functions
			m_minConstraint, m_maxConstraint, true  // Node output constraints
		);
		m_CGPOutputDendriteUpdater->randomize();
	}
	
	void JBrain::createAxonUpdater()
	{
		m_CGPAxonUpdater = new CGP::JBrainCGPIndividual(
			static_cast<unsigned int>(m_axonInputs.size()), // Number of inputs
			static_cast<unsigned int>(m_axonOutputs.size()), // Number of outputs
			1, // Number of rows
			m_axonProgramNodes, // Number of columns
			m_axonProgramNodes, // Columns back to search
			m_minP, m_maxP, // P-values for CGP functions
			m_functionStringList, // Available functions as strings
			m_functionList, // Available functions as callable (dbl, dbl, dbl) functions
			m_minConstraint, m_maxConstraint, true  // Node output constraints
		);
		m_CGPAxonUpdater->randomize();
	}

	void JBrain::createNeuronUpdater()
	{
		m_CGPNeuronUpdater = new CGP::JBrainCGPIndividual(
			static_cast<unsigned int>(m_neuronInputs.size()), // Number of inputs
			static_cast<unsigned int>(m_neuronOutputs.size()), // Number of outputs
			1, // Number of rows
			m_neuronProgramNodes, // Number of columns
			m_neuronProgramNodes, // Columns back to search
			m_minP, m_maxP, // P-values for CGP functions
			m_functionStringList, // Available functions as strings
			m_functionList, // Available functions as callable (dbl, dbl, dbl) functions
			m_minConstraint, m_maxConstraint, true  // Node output constraints
		);
		m_CGPNeuronUpdater->randomize();
	}

	void JBrain::createChemicalUpdater()
	{
		// Not implemented, yet:
		m_CGPChemicalUpdater = nullptr;
		// m_CGPChemicalUpdater->randomize();
	}

	void JBrain::createAllSeparateUpdaters()
	{
		createDendriteUpdater();
		createOutputDendriteUpdater();
		createAxonUpdater();
		createNeuronUpdater();
		createChemicalUpdater();
	}

	unsigned int JBrain::getNextNeuronNumber()
	{
		return m_currNeuronNumber++;
	}

	void JBrain::handleBrainSizeChange()
	{
		// If one size changed, but we're enforcing cube-shaped brains, make the change:
		if (m_brainUseSameDimensions)
		{
			if ((m_brainXSize != m_brainYSize) || (m_brainYSize != m_brainZSize))
			{
				float newBrainSize = std::max(m_brainXSize, std::max(m_brainYSize, m_brainZSize));
				m_brainXSize = m_brainYSize = m_brainZSize = newBrainSize;
			}
		}

		// Constrain the location of all parts of the brain:
		for (auto& neuron : m_neurons)
		{
			neuron.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);
			for (auto& axon : neuron.m_axons)
			{
				axon.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);
				axon.constrainLength(neuron.m_X, neuron.m_Y, neuron.m_Z, m_axonMaxLength);
			}

			for (auto& dendrite : neuron.m_dendrites)
			{
				dendrite.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);
				dendrite.constrainLength(neuron.m_X, neuron.m_Y, neuron.m_Z, m_dendriteMaxLength);
			}
		}

		// Constrain the location of our inputs and outputs:
		for (auto& axon : m_inputAxons)
			axon.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);

		for (auto& dendrite : m_outputDendrites)
			dendrite.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);

		// Finally, if our inputs or outputs are expected to be on the edge of the
		// brain, make sure they still are:
		if (m_brainInputsOnOneSide)
		{
			// This shouldn't change since 0.0 is the minimum brain location,
			// just being safe:
			for (auto& axon : m_inputAxons)
				axon.m_X = 0.0;

			for (auto& dendrite : m_outputDendrites)
				dendrite.m_X = m_brainXSize;
		}
	}

	float JBrain::getRandomFloat(const float& min, const float& max)
	{
		// Make sure we don't hit bad argument issues:
		float useMin = min;
		float useMax = max;
		if (min >= max)
		{
			useMin = max;
			useMax = min;
		}

		// Random device and distribution don't need to be
		// recreated every time:
		static std::random_device rd;
		static std::mt19937 e2(rd());

		// We always want uniform distribution. The odd next-after
		// syntax around max is used to make sure that max is one
		// of the values that can be returned. The distribution's possible
		// return values are in the range [a, b):
		std::uniform_real_distribution<> dist(useMin,
			std::nextafter(useMax, std::numeric_limits<float>::max()));

		return static_cast<float>(dist(e2));
	}

	int JBrain::getRandomInt(const int& min, const int& max)
	{
		// Random device and distribution don't need to be
		// recreated every time:
		static std::random_device rd;
		static std::mt19937 e2(rd());

		std::uniform_int_distribution<> dist(min, max);

		return dist(e2);
	}

	float JBrain::getDistance(const JBrainComponent& a, const JBrainComponent& b)
	{
		return getDistance(a.m_X, a.m_Y, a.m_Z, b.m_X, b.m_Y, b.m_Z);
	}

	float JBrain::getDistance(const float& x1, const float& y1, const float& z1,
		const float& x2, const float& y2, const float& z2)
	{
		// I wish we had a good distance formula that didn't require
		// square roots and multiplication:
		float dx = x1 - x2;
		float dy = y1 - y2;
		float dz = z1 - z2;

		// X^0.5 = sqrt(X):
		return static_cast<float>(pow((dx * dx) + (dy * dy) + (dz * dz), 0.5));
	}

	void JBrain::applyDistanceDeterioration(float& fireValue, const float& distance)
	{
		fireValue = fireValue / (1 + (m_neuronFireSpaceDeterioration * (distance * distance)));
	}

	void JBrain::applyTimeDeterioration(float& fireValue, const float& time)
	{
		fireValue = fireValue / (1 + (m_neuronFireTimeDeterioration * time));
	}

	void JBrain::fireAllNeurons()
	{
		for (unsigned int i = 0; i < m_neurons.size(); ++i)
		{
			if (getIfNeuronFires(m_neurons[i]))
			{
				// Fired? Change its fire threshold:
				if (m_neuronUseDynamicFireThresholds)
					m_neurons[i].m_fireThreshold += m_neuronFireThresholdActiveChange;

				// Add a neuron neuron-fired event at every axon output:
				for (unsigned int j = 0; j < m_neurons[i].m_axons.size(); ++j)
				{
					m_neuronFires.push_back(
						NeuronFire(m_neurons[i].m_axons[j].m_X,
							m_neurons[i].m_axons[j].m_Y,
							m_neurons[i].m_axons[j].m_Z,
							m_neurons[i].m_fireValue,
							false));
				}
				
				// Set it as fired on this time step.
				// This value is set to -1 because it will be incremented
				// before doing any checks on the next time step.
				m_neurons[i].m_timeStepsSinceLastFire = -1;
			}
			else // Didn't fire? Change its fire threshold:
			{
				if (m_neuronUseDynamicFireThresholds)
					m_neurons[i].m_fireThreshold += m_neuronFireThresholdIdleChange;
			}
		}
	}

	void JBrain::createInputsAndOutputs()
	{
		// Inputs can either be on one side or placed randomly
		// throughout the brain. If single-side is selected, the
		// inputs will be on the X=0 side of the cuboid brain. If outputs
		// are on a single side, they will be on the X = max(X) side of
		// the cuboid.
		float inX, inY, inZ;
		for (unsigned int i = 0; i < m_observationSize; ++i)
		{
			inY = getRandomFloat(0.0, m_brainYSize);
			inZ = getRandomFloat(0.0, m_brainZSize);
			if (m_brainInputsOnOneSide)
				inX = 0.0;
			else
				inX = getRandomFloat(0.0, m_brainXSize);

			m_inputAxons.push_back(JAxon(inX, inY, inZ));
		}

		m_outputDendrites.clear();
		m_outputNeurons.clear();

		// Outputs on one side or randomly placed:
		for (unsigned int i = 0; i < m_actionSize; ++i)
		{
			inY = getRandomFloat(0.0, m_brainYSize);
			inZ = getRandomFloat(0.0, m_brainZSize);

			if (m_brainOutputsOnOneSide)
				inX = m_brainXSize;
			else
				inX = getRandomFloat(0.0, m_brainXSize);

			// Create an output neuron. (-1, 0) -> rand dendrite, zero axons:
			if (m_useOutputNeurons)
				m_outputNeurons.push_back(createNewNeuron(inX, inY, inZ, -1, 0));
			else  // Create an output dendrite, default to 1.0 weight:
				m_outputDendrites.push_back(JDendrite(inX, inY, inZ, 1.0));
		}
	}

	void JBrain::setAllInputAxonFires(const std::vector<double>& inputs)
	{
		// Create a firing event at each input axon corresponding to the
		// value of the inputs.

		// We should get inputs of the exact size of input axons we have:
		assert(inputs.size() == m_inputAxons.size());

		for (unsigned int i = 0; i < inputs.size(); ++i)
		{
			NeuronFire tempNF = NeuronFire(
				m_inputAxons[i].m_X, // X
				m_inputAxons[i].m_Y, // Y
				m_inputAxons[i].m_Z, // Z
				static_cast<float>(inputs[i]), // fireValue
				true);  // environmentInput

			// Environment Inputs may not age, so we need to manually
			// set the age to 0 to make the time-deterioration equations
			// avoid error:
			if (!m_inputNeuronFiresAge)
				tempNF.m_age = 0;

			// Add it to our list of neuron firings:
			m_neuronFires.push_back(tempNF);
		}
	}

	std::vector<double> JBrain::processInput(const std::vector<double>& inputs,
		int sageChoice, bool newTrial)
	{
		// If this is the first observation of a new run, record the inputs
		// and reset some variables:
		if (newTrial)
		{
			m_initialObservation = inputs;
			m_totalTrialInputsProcessed = 0;
			m_totalTrialSageChoiceMatches = 0;

			for (JNeuron& neuron : m_neurons)
			{
				neuron.m_fireOpportunitiesInThisRun = 0;
				neuron.m_timesFiredInThisRun = 0;
			}
		}

		// Check to see if we should reset, and do it if so:
		resetBrainForNewInputs();
				
		// Take the environment inputs and set them as the brain's inputs:
		setAllInputAxonFires(inputs);

		// Create our vector of vectors to average all outputs together:
		std::vector<std::vector<double> > allOutputs;

		// Make sure we don't roll over the unsigined int when determining when to start
		// recording brain outputs:
		unsigned int startOutputRecordStep = 0;
		if (m_brainProcessingStepsBetweenInputAndOutput > m_brainOutputsToAverageTogether)
			startOutputRecordStep = m_brainProcessingStepsBetweenInputAndOutput - m_brainOutputsToAverageTogether;

		// Take an appropriate number of steps forward:
		for (unsigned int i = 0; i < m_brainProcessingStepsBetweenInputAndOutput; ++i)
		{
			singleTimeStepForward();

			// Check brain output:
			if (i >= startOutputRecordStep)
			{
				allOutputs.push_back(readBrainOutputs());
			}
		}
	
		// Average all outputs together (there should be at least 1):
		m_mostRecentBrainOutput = allOutputs[0];
		for (unsigned int i = 1; i < allOutputs.size(); ++i)
			for (unsigned int j = 0; j < allOutputs[i].size(); ++j)
				m_mostRecentBrainOutput[j] += allOutputs[i][j];

		for (unsigned int i = 0; i < m_mostRecentBrainOutput.size(); ++i)
			m_mostRecentBrainOutput[i] /= static_cast<double>(allOutputs.size());

		int brainChoice = static_cast<int>(std::distance(
			m_mostRecentBrainOutput.begin(),
			  std::max_element(m_mostRecentBrainOutput.begin(),
				m_mostRecentBrainOutput.end())));
		
		if (sageChoice >= 0)  // We care about the sage choice:
		{
			m_sageChoices.push_back(sageChoice);

			// Distance between the beginning of the vector and the
			// location of the max value in the vector:
			m_brainChoices.push_back(brainChoice);
		}

		// Record full-run data:
		++m_totalTrialInputsProcessed;
		if (brainChoice == sageChoice)
			++m_totalTrialSageChoiceMatches;

		updateAfterProcessingInput();

		return m_mostRecentBrainOutput;
	}

	void JBrain::singleTimeStepForward()
	{
		incrementAllNeuronFireAges();
		allNeuronsSingleTimeStepForward();
		fireAllNeurons();
	}

	void JBrain::calculateSageMatch()
	{
		assert(m_sageChoices.size() == m_brainChoices.size());
		m_sageMatchPercent = 1.0;

		// No decisions made yet?
		if (m_sageChoices.size() != 0)
		{
			unsigned int correctCount = 0;
			for (unsigned int i = 0; i < m_sageChoices.size(); ++i)
			{
				if (m_sageChoices[i] == m_brainChoices[i])
					++correctCount;
			}

			// Set the percent (0.0 - 1.0) of the time our answers matched:
			m_sageMatchPercent = static_cast<float>(correctCount) / static_cast<float>(m_sageChoices.size());
		}
	}

	void JBrain::calculateNearestDendriteToEveryAxon()
	{
		// We only need to worry about dendrites/axons attached to neurons
		// (not inputs or outputs). So, we can just loop through all
		// neurons.
		float minDist;
		float dist;
		int minDistNeuronIdx = -1;
		int minDistDendriteIdx = -1;
		int minDistEnvIdx = -1;
		float dendriteX;
		float dendriteY;
		float dendriteZ;

		for (unsigned int n1 = 0; n1 < m_neurons.size(); ++n1)
		{
			for (unsigned int ax = 0; ax < m_neurons[n1].m_axons.size(); ++ax)
			{
				minDist = 100000000.0f;
				minDistNeuronIdx = -1;
				for (unsigned int n2 = 0; n2 < m_neurons.size(); ++n2)
				{
					for (unsigned int den = 0; den < m_neurons[n2].m_dendrites.size(); ++den)
					{
						dist = getDistance(m_neurons[n1].m_axons[ax],
							m_neurons[n2].m_dendrites[den]);
						if (dist < minDist)
						{
							minDist = dist;
							minDistNeuronIdx = n2;
							minDistDendriteIdx = den;
						}
					} // end for all dendrites in neuron 2
				} // end loop through target neurons

				assert(minDistDendriteIdx != -1);

				// Loop through the action (output) dendrites:
				minDistEnvIdx = -1;
				for (unsigned int i = 0; i < m_outputDendrites.size(); ++i)
				{
					dist = getDistance(m_neurons[n1].m_axons[ax],
						m_outputDendrites[i]);

					if (dist < minDist)
					{
						minDistEnvIdx = i;
						minDist = dist;
					}
				}

				// If we didn't find an environment axon closer, use the nearest
				// standard axon:
				if (minDistEnvIdx == -1)
				{
					dendriteX = m_neurons[minDistNeuronIdx].m_dendrites[minDistDendriteIdx].m_X;
					dendriteY = m_neurons[minDistNeuronIdx].m_dendrites[minDistDendriteIdx].m_Y;
					dendriteZ = m_neurons[minDistNeuronIdx].m_dendrites[minDistDendriteIdx].m_Z;
					m_neurons[n1].m_axons[ax].m_nearestDendriteIsActionDendrite =
						false;

					m_neurons[n1].m_axons[ax].m_nearestDendriteIsPartOfSameNeuron =
						(n1 == minDistNeuronIdx);
				}
				else  // Use the action dendrite:
				{
					dendriteX = m_outputDendrites[minDistEnvIdx].m_X;
					dendriteY = m_outputDendrites[minDistEnvIdx].m_Y;
					dendriteZ = m_outputDendrites[minDistEnvIdx].m_Z;
					m_neurons[n1].m_axons[ax].m_nearestDendriteIsActionDendrite =
						true;

					m_neurons[n1].m_axons[ax].m_nearestDendriteIsPartOfSameNeuron =
						false;
				}

				// Set the distance/direction to the nearest axon:
				m_neurons[n1].m_axons[ax].m_nearestDendriteX =
					dendriteX - m_neurons[n1].m_axons[ax].m_X;

				m_neurons[n1].m_axons[ax].m_nearestDendriteY =
					dendriteY - m_neurons[n1].m_axons[ax].m_Y;

				m_neurons[n1].m_axons[ax].m_nearestDendriteZ =
					dendriteZ - m_neurons[n1].m_axons[ax].m_Z;

				m_neurons[n1].m_axons[ax].m_nearestDendriteDistance = minDist;
			}
		}
	}

	void JBrain::calculateNearestAxonToEveryDendrite()
	{
		// We only need to worry about dendrites/axons attached to neurons
		// (not inputs or outputs). So, we can just loop through all
		// neurons.
		float minDist;
		float dist;
		int minDistNeuronIdx = -1;
		int minDistAxonIdx = -1;
		int minDistEnvIdx = -1;
		float axonX;
		float axonY;
		float axonZ;

		std::vector<float> DEBUG_DISTANCES;

		for (unsigned int n1 = 0; n1 < m_neurons.size(); ++n1)
		{
			for (unsigned int den = 0; den < m_neurons[n1].m_dendrites.size(); ++den)
			{
				minDist = 10000000.0f;
				minDistNeuronIdx = -1;

				for (unsigned int n2 = 0; n2 < m_neurons.size(); ++n2)
				{
					for (unsigned int ax = 0; ax < m_neurons[n2].m_axons.size(); ++ax)
					{
						dist = getDistance(m_neurons[n1].m_dendrites[den],
							m_neurons[n2].m_axons[ax]);
						
						if (dist < minDist)
						{
							minDist = dist;
							minDistNeuronIdx = n2;
							minDistAxonIdx = ax;
						}
					} // End for all axons on second-loop neuron
				} // End loop through target neurons

				// Set the values for this dendrite. Direction = location of target - self location:
				assert(minDistNeuronIdx != -1);

				// For this dendrite, loop through the environment axons and
				// see if one is closer:
				minDistEnvIdx = -1;
				for (unsigned int i = 0; i < m_inputAxons.size(); ++i)
				{
					dist = getDistance(m_neurons[n1].m_dendrites[den],
						m_inputAxons[i]);

					if (dist < minDist)
					{
						minDist = dist;
						minDistEnvIdx = i;
					}
				}

				// If we didn't find an environment axon closer, use the nearest
				// standard axon: 
				if (minDistEnvIdx == -1)
				{
					axonX = m_neurons[minDistNeuronIdx].m_axons[minDistAxonIdx].m_X;
					axonY = m_neurons[minDistNeuronIdx].m_axons[minDistAxonIdx].m_Y;
					axonZ = m_neurons[minDistNeuronIdx].m_axons[minDistAxonIdx].m_Z;
					m_neurons[n1].m_dendrites[den].m_nearestAxonIsEnvironmentAxon =
						false;

					m_neurons[n1].m_dendrites[den].m_nearestAxonPartOfSameNeuron =
						(n1 == minDistNeuronIdx);
				}
				else  // Use the env axon:
				{
					axonX = m_inputAxons[minDistEnvIdx].m_X;
					axonY = m_inputAxons[minDistEnvIdx].m_Y;
					axonZ = m_inputAxons[minDistEnvIdx].m_Z;
					m_neurons[n1].m_dendrites[den].m_nearestAxonIsEnvironmentAxon =
						true;

					m_neurons[n1].m_dendrites[den].m_nearestAxonPartOfSameNeuron = 
						false;
				}
				
				// Set the distance/direction to the nearest axon:
				m_neurons[n1].m_dendrites[den].m_nearestAxonX =
					axonX - m_neurons[n1].m_dendrites[den].m_X;

				m_neurons[n1].m_dendrites[den].m_nearestAxonY =
					axonY - m_neurons[n1].m_dendrites[den].m_Y;

				m_neurons[n1].m_dendrites[den].m_nearestAxonZ =
					axonZ - m_neurons[n1].m_dendrites[den].m_Z;

				m_neurons[n1].m_dendrites[den].m_nearestAxonDistance = minDist;

			} // End loop through axons of source neuron
		} // End loop through source neurons
	}

	void JBrain::updateAfterProcessingInput()
	{
		// We only worry about doing any updating if brain-output
		// is our update signal:
		if (m_updateEvent == CGP::UPDATE_EVENT::BRAIN_OUTPUT)
		{
			++m_inputProcessingsSinceLastUpdate;
			// With an update frequency of 1, sageMatchPercent will always
			// be either 1.0 or 0.0.
			if (m_inputProcessingsSinceLastUpdate >= m_updateFrequency)
			{
				// Config.yaml name: SAGE_MATCH_PERCENT
				calculateSageMatch();

				// Config.yaml name: NEAREST_AXON_...
				calculateNearestAxonToEveryDendrite();

				// Make sure we know how often neurons fired:
				calculateAverageNeuronFirePercentage();

				// Reset for next update event:
				m_inputProcessingsSinceLastUpdate = 0;
				m_sageChoices.clear();
				m_brainChoices.clear();

				// Every neuron needs to have their times-fired-count
				// variables reset:
				for (JNeuron& neuron : m_neurons)
				{
					neuron.m_fireOpportunitiesSinceLastUpdate = 0;
					neuron.m_timesFiredSinceLastUpdate = 0;
				}

				applyAllCGP();
				
				// All health values have been updated, check for duplications and deaths:
				duplicateAndKillNeurons();
			}
		}
	}

	void JBrain::duplicateAndKillNeurons()
	{
		// Go through the neurons backwards so that neurons added don't
		// get checked:

		// Track (duplications - deaths):
		int duplicationDeaths = 0;
		// Odd loop syntax because unsigned ints loop around after zero:
		for (int i = static_cast<int>(m_neurons.size() - 1); i >= 0; --i)
		{
			if (m_neurons[i].m_health > m_neuronDuplicateHealth)
			{
				// Only create another neuron if we aren't at our limit:
				if (m_neurons.size() < m_maxNeurons)
					m_neurons.push_back(createDuplicateNeuron(m_neurons[i]));
				
				// Keep raising the creation threshold in any case:
				++duplicationDeaths;
			}
			else if (m_neurons[i].m_health < m_neuronDeathHealth)
			{
				m_neurons.erase(m_neurons.begin() + i);
				--duplicationDeaths;
			}
		}

		// Multiply with more duplications than deaths, divide otherwise:
		while (duplicationDeaths > 0)
		{
			m_neuronDuplicateHealth *= m_neuronDeathDuplicateHealthThresholdMultiplier;
			m_neuronDeathHealth /= m_neuronDeathDuplicateHealthThresholdMultiplier;
			--duplicationDeaths;
		}

		while (duplicationDeaths < 0)
		{
			m_neuronDuplicateHealth /= m_neuronDeathDuplicateHealthThresholdMultiplier;
			m_neuronDeathHealth *= m_neuronDeathDuplicateHealthThresholdMultiplier;
			++duplicationDeaths;
		}
	}

	JNeuron JBrain::createDuplicateNeuron(JNeuron& neuron)
	{
		float finalX = neuron.m_X;
		float finalY = neuron.m_Y;
		float finalZ = neuron.m_Z;

		// Change the starting neuron's health due to being duplicated:
		neuron.m_health += m_neuronDuplicationHealthChange;

		if (m_neuronDuplicateNearby)
		{
			// If the neuron needs to be nearby, determine the coordinates
			// based on the duplicated neuron:
			float xChange = getRandomFloat(m_neuronMinNearbyDistance, m_neuronMaxNearbyDistance);
			float yChange = getRandomFloat(m_neuronMinNearbyDistance, m_neuronMaxNearbyDistance);
			float zChange = getRandomFloat(m_neuronMinNearbyDistance, m_neuronMaxNearbyDistance);
		
			// Add or subtract the change, randomly for each coordinate:
			if (getRandomInt(0, 1) == 0)
				finalX -= xChange;
			else
				finalX += xChange;

			if (getRandomInt(0, 1) == 0)
				finalY -= yChange;
			else
				finalY += yChange;

			if (getRandomInt(0, 1) == 0)
				finalZ -= zChange;
			else
				finalZ += zChange;

			// Keep everyone in bounds:
			finalX = static_cast<float>(fmin(fmax(0.0, finalX), m_brainXSize));
			finalY = static_cast<float>(fmin(fmax(0.0, finalY), m_brainYSize));
			finalZ = static_cast<float>(fmin(fmax(0.0, finalZ), m_brainZSize));			
		}
		else
		{
			// Not nearby, just select randomly:
			finalX = getRandomFloat(0.0, m_brainXSize);
			finalY = getRandomFloat(0.0, m_brainYSize);
			finalZ = getRandomFloat(0.0, m_brainZSize);
		}

		return createNewNeuron(finalX, finalY, finalZ);
	}

	void JBrain::resetBrainForNewInputs()
	{
		// Clear the most recent brain output. Not strictly necessary since it will
		// be overwritten by the processing of input, but want to catch bugs:
		m_mostRecentBrainOutput.clear();

		// If we are allowed to do a full reset, it is easy:
		if (m_brainResetBeforeProcessingInput)
		{
			// Clear all neuron fires:
			m_neuronFires.clear();

			// Set all neurons to ready-to-fire state:
			for (JNeuron& neuron : m_neurons)
				neuron.m_timeStepsSinceLastFire = m_neuronRefractoryPeriod + 1;
				
		}
		// Can't do a full reset, but still need to remove the previous inputs:
		else
		{
			std::vector<unsigned int> idx;
			for (unsigned int i = 0; i < m_neuronFires.size(); ++i)
				if (m_neuronFires[i].m_environmentInput)
					idx.push_back(i);

			// Remove them in reverse order so the correct values are
			// removed:
			for (auto it = idx.rbegin(); it != idx.rend(); ++it)
				m_neuronFires.erase(m_neuronFires.begin() + *it);			 
		}
	}

	float JBrain::applyJNeuronActivationFunction(const float& input)
	{
		float retVal = input;
		static float e = 2.718281828459045235360284171352f;

		// Ignore the "none" case:
		switch (m_jNeuronActivationFunction)
		{
		case CGP::JNEURON_ACTIVATION_FUNCTION::SIGMOID:
			retVal = 1.0f / (1.0f + powf(e, -retVal));
			break;

		case CGP::JNEURON_ACTIVATION_FUNCTION::TANH:
			retVal = tanhf(input);
			break;
		}

		return retVal;
	}

	void JBrain::incrementAllNeuronFireAges()
	{
		// Better way would be to define a "tooOld" lambda function and
		// let C++ handle this, but hitting issues passing the constant
		// too-old-age in.
		std::vector<unsigned int> tooOldIdx;		
		for (unsigned int i = 0; i < m_neuronFires.size(); ++i)
		{
			// Inputs from the environment may not get their ages incremented:
			if (m_inputNeuronFiresAge || !m_neuronFires[i].m_environmentInput)
			{
				++m_neuronFires[i].m_age;
				if (m_neuronFires[i].m_age > static_cast<int>(m_neuronFireLifetime))
					tooOldIdx.push_back(i);
			}
		}

		// Remove in reverse order so that the correct values
		// are removed:
		for (auto it = tooOldIdx.rbegin(); it != tooOldIdx.rend(); ++it)
		{
			m_neuronFires.erase(m_neuronFires.begin() + *it);
		}
	}

	void JBrain::allNeuronsSingleTimeStepForward()
	{
		for (JNeuron& neuron : m_neurons)
		{
			++neuron.m_timeStepsSinceLastFire;
			++neuron.m_age;
		}
	}

	float JBrain::calculateInternalNeuronValue(JNeuron& neuron, const bool& ignoreEnvironmentInputs)
	{
		float totalDendriteInputs = 0.0;
		for (unsigned int i = 0; i < neuron.m_dendrites.size(); ++i)
		{
			totalDendriteInputs += getDendriteInput(neuron.m_dendrites[i], ignoreEnvironmentInputs);
		}

		totalDendriteInputs = applyJNeuronActivationFunction(totalDendriteInputs);
		return totalDendriteInputs;
	}

	bool JBrain::getIfNeuronFires(JNeuron& neuron)
	{
		bool retFire = false;

		// If we aren't using dynamic firing thresholds, the neuron needs to be
		// out of its refractory period:
		if (m_neuronUseDynamicFireThresholds ||
			neuron.m_timeStepsSinceLastFire >= static_cast<int>(m_neuronRefractoryPeriod))
		{
			// This was an opportunity to fire, mark it as such:
			++neuron.m_fireOpportunitiesSinceLastUpdate;
			++neuron.m_fireOpportunitiesInThisRun;

			float totalDendriteInputs = calculateInternalNeuronValue(neuron);

			// Neuron firing is either probabilistic or based on a threshold value
			// being exceeded. Probabilistic still uses the dendrite inputs, but
			// as part of a random variable selection:
			if (m_neuronProbabilisticFire &&
				getRandomFloat(0.0, totalDendriteInputs) >= neuron.m_fireThreshold)
				retFire = true;								
			else if (totalDendriteInputs >= neuron.m_fireThreshold)
				retFire = true;
		}

		// If we fired, update some variables as such:
		if (retFire)
		{
			neuron.m_timeStepsSinceLastFire = 0;
			++neuron.m_timesFiredSinceLastUpdate;
			++neuron.m_timesFiredInThisRun;
		}

		return retFire;
	}

	float JBrain::getDendriteInput(JDendrite& dendrite, const bool& ignoreEnvironmentInputs)
	{
		// Go through every neuron firing and calculate its input into this dendrite:
		float distance;		
		float fireValue;
		float maxFireValue = 0.0;
		unsigned int maxFireIndex = -1;
		float maxFireDistance = 0.0;
		float totalInput = 0.0;
		for (unsigned int i = 0; i < m_neuronFires.size(); ++i)
		{
			// Environment input and we should ignore it:
			if (m_neuronFires[i].m_environmentInput && ignoreEnvironmentInputs)
				continue;

			// Every neuron fires at a specific value:
			fireValue = m_neuronFires[i].m_fireValue;

			// It changes based on how far away the listening dendrite is:
			distance = getDistance(dendrite, m_neuronFires[i]);
			applyDistanceDeterioration(fireValue, distance);

			// It changes based on how long since it fired:
			applyTimeDeterioration(fireValue, static_cast<float>(m_neuronFires[i].m_age));

			// Add to this dendrite's total input:
			totalInput += fireValue;

			// Check if we found a new favorite:
			if (fireValue > maxFireValue)
			{
				maxFireValue = fireValue;
				maxFireIndex = i;
				maxFireDistance = distance;
			}
		}

		// If there was at least some input to this dendrite:
		if (totalInput > 0.0)
		{
			dendrite.m_biggestInputX = m_neuronFires[maxFireIndex].m_X - dendrite.m_X;
			dendrite.m_biggestInputY = m_neuronFires[maxFireIndex].m_Y - dendrite.m_Y;
			dendrite.m_biggestInputZ = m_neuronFires[maxFireIndex].m_Z - dendrite.m_Z;
			dendrite.m_biggestInputDistance = maxFireDistance;
			dendrite.m_recentInputAvailable = true;
			dendrite.m_biggestInputValue = maxFireValue;
			dendrite.m_currentValue = totalInput * dendrite.m_weight;
			dendrite.m_biggestInputIsEnvironmentAxon = m_neuronFires[maxFireIndex].m_environmentInput;
		}
		// No input, mark as such:
		else
		{
			dendrite.m_biggestInputX = dendrite.m_biggestInputY = dendrite.m_biggestInputZ = 0.0;
			dendrite.m_recentInputAvailable = false;
			dendrite.m_currentValue = 0.0;
		}

		return dendrite.m_currentValue;
	}

	// Go through the disembodied dendrites that represent the brain's
	// choice-outputs, read the values there, and return the resulting
	// vector of doubles:
	std::vector<double> JBrain::readBrainOutputs()
	{
		std::vector<double> retVal;

		// Output neurons:
		if (m_useOutputNeurons)
		{
			for (unsigned int i = 0; i < m_outputNeurons.size(); ++i)
			{
				retVal.push_back(calculateInternalNeuronValue(m_outputNeurons[i], m_brainOutputsIgnoreEnvironmentInputs));
			}
		}
		else // Output dendrites:
		{
			for (unsigned int i = 0; i < m_outputDendrites.size(); ++i)
			{
				retVal.push_back(getDendriteInput(m_outputDendrites[i], m_brainOutputsIgnoreEnvironmentInputs));
			}
		}

		return retVal;
	}

	void JBrain::calculateAverageNeuronFirePercentage()
	{
		// Get the average brain fire to start with:
		m_averageNeuronFirePercentage = 0.0;
		for (JNeuron& neuron : m_neurons)
			m_averageNeuronFirePercentage += static_cast<float>(neuron.getPercentageFire());

		m_averageNeuronFirePercentage /= static_cast<float>(m_neurons.size());		
	}

	void JBrain::applyCGP(JNeuron& neuron)
	{
		std::vector<double> nCGPInputs = getCGPInputs(neuron);
		std::vector<double> nCGPOutputs = m_CGPNeuronUpdater->
			calculateOutputs(nCGPInputs);
		applyCGPOutputs(neuron, nCGPOutputs);
	}

	void JBrain::applyCGP(JDendrite& dendrite, const unsigned int& parentNeuronNum, const bool& outputNeuron)
	{
		std::vector<double> dCGPInputs = getCGPInputs(dendrite, parentNeuronNum, outputNeuron);
		std::vector<double> dCGPOutputs;
		// Apply a different CGP program, depending on the type of neuron:
		if (outputNeuron)
			dCGPOutputs = m_CGPOutputDendriteUpdater->calculateOutputs(dCGPInputs);
		else
			dCGPOutputs = m_CGPDendriteUpdater->calculateOutputs(dCGPInputs);

		applyCGPOutputs(dendrite, dCGPOutputs, parentNeuronNum, outputNeuron);
	}

	void JBrain::applyCGP(JAxon& axon, const JNeuron& parentNeuron)
	{
		std::vector<double> aCGPInputs = getCGPInputs(axon, parentNeuron);
		std::vector<double> aCGPOutputs = m_CGPAxonUpdater->
			calculateOutputs(aCGPOutputs);
		applyCGPOutputs(axon, aCGPOutputs, parentNeuron);
	}

	void JBrain::applyCGPOutputs(JNeuron& neuron, const std::vector<double>& cgpOutputs)
	{
		// There should only be one output:
		assert(cgpOutputs.size() == 1);

		switch (m_neuronOutputs[0])
		{
			case CGP::CGP_OUTPUT::HEALTH:
				neuron.m_health += static_cast<float>(cgpOutputs[0]);
				break;

			case CGP::CGP_OUTPUT::HEALTH_INCREASE_DECREASE:
				// Output value indicates increase or decrease:
				if (cgpOutputs[0] < m_neuronCGPOutputLowHealthChange)
					neuron.m_health -= m_neuronCGPOutputHealthChangeAmount;
				else if (cgpOutputs[0] > m_neuronCGPOutputHighHealthChange)
					neuron.m_health += m_neuronCGPOutputHealthChangeAmount;
				// else:
					// No change
				break;
		}
	}

	void JBrain::applyCGPOutputs(JAxon& axon, const std::vector<double>& cgpOutputs,
		const JNeuron& parentNeuron)
	{
		// Track the current output being used:
		unsigned int cgpIdx = 0;

		for (unsigned int i = 0; i < cgpOutputs.size(); ++i)
		{
			switch (m_dendriteOutputs[i])
			{
			case CGP::CGP_OUTPUT::LOCATION:
				// Location changes are broken into 3 values, one each for 
				// X, Y, and Z:
				axon.m_X += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				axon.m_Y += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				axon.m_Z += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::RANDOM_MOVEMENT_THRESHOLD:
				// Random movement uses the highMoveToward threshold:
				if (cgpOutputs[cgpIdx] > m_axonHighMoveToward)
				{
					axon.m_X += getRandomFloat(-m_axonAwayTowardMoveAmount, m_axonAwayTowardMoveAmount);
					axon.m_Y += getRandomFloat(-m_axonAwayTowardMoveAmount, m_axonAwayTowardMoveAmount);
					axon.m_Z += getRandomFloat(-m_axonAwayTowardMoveAmount, m_axonAwayTowardMoveAmount);
				}
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::HEALTH:
				std::cout << "Axon health in CGP output params. Not yet implemented." << std::endl;
				++cgpIdx;
				break;

			// If this value is lower than a set value or higher than a different
			// set value, the axon moves a static amount. Otherwise, there is no movement:
			case CGP::CGP_OUTPUT::NEAREST_DENDRITE_CLOSER_FURTHER:
				if (cgpOutputs[cgpIdx] < m_axonLowMoveAway)
				{
					axon.m_X -= (axon.m_nearestDendriteX - axon.m_X) * m_axonAwayTowardMoveAmount;
					axon.m_Y -= (axon.m_nearestDendriteY - axon.m_Y) * m_axonAwayTowardMoveAmount;
					axon.m_Z -= (axon.m_nearestDendriteZ - axon.m_Z) * m_axonAwayTowardMoveAmount;
				}
				else if (cgpOutputs[cgpIdx] > m_axonHighMoveToward)
				{
					axon.m_X += (axon.m_nearestDendriteX - axon.m_X) * m_axonAwayTowardMoveAmount;
					axon.m_Y += (axon.m_nearestDendriteY - axon.m_Y) * m_axonAwayTowardMoveAmount;
					axon.m_Z += (axon.m_nearestDendriteZ - axon.m_Z) * m_axonAwayTowardMoveAmount;
				}
				// else: No movement.
				++cgpIdx;
				break;
			}
		}
		// Ensure we stay in a valid location:
		axon.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);
		axon.constrainLength(parentNeuron.m_X, parentNeuron.m_Y, parentNeuron.m_Z,
			m_axonMaxLength);
	}

	void JBrain::applyCGPOutputs(JDendrite& dendrite, const std::vector<double>& cgpOutputs,
		const unsigned int& parentNeuronNum, const bool& outputNeuron)
	{
		// Sometimes, a dendrite output may take more than one cgpOutput.
		unsigned int cgpIdx = 0;
		float moveVal;
		JNeuron parentNeuron = outputNeuron ? m_outputNeurons[parentNeuronNum] :
			m_neurons[parentNeuronNum];

		for (unsigned int i = 0; i < cgpOutputs.size(); ++i)
		{
			switch (m_dendriteOutputs[i])
			{
			case CGP::CGP_OUTPUT::LOCATION:
				// Location changes are broken into 3 values, one each for 
				// X, Y, and Z:
				dendrite.m_X += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				dendrite.m_Y += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				dendrite.m_Z += static_cast<float>(cgpOutputs[cgpIdx]);
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::CLOSER_TO_STRONGEST_INPUT:
				// Get closer to the strongest input by <value> percent.
				// Negative values can move it further away. This value
				// is locked between -1 and 1:
				moveVal = static_cast<float>(cgpOutputs[cgpIdx]);
				moveVal = static_cast<float>(fmin(fmax(-1.0, moveVal), 1.0));
				dendrite.m_X += (dendrite.m_biggestInputX - dendrite.m_X) * moveVal;
				dendrite.m_Y += (dendrite.m_biggestInputY - dendrite.m_Y) * moveVal;
				dendrite.m_Z += (dendrite.m_biggestInputZ - dendrite.m_Z) * moveVal;
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::CLOSER_TO_NEAREST_AXON:
				// Get closer to the nearest axon by <value> percent.
				// Negative values can move it further away.
				moveVal = static_cast<float>(cgpOutputs[cgpIdx]);
				dendrite.m_X += (dendrite.m_nearestAxonX - dendrite.m_X) * moveVal;
				dendrite.m_Y += (dendrite.m_nearestAxonY - dendrite.m_Y) * moveVal;
				dendrite.m_Z += (dendrite.m_nearestAxonZ - dendrite.m_Z) * moveVal;
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::RANDOM_MOVEMENT_THRESHOLD:
				// Random movement uses the highMoveToward threshold:
				if (cgpOutputs[cgpIdx] > m_axonHighMoveToward)
				{
					dendrite.m_X += getRandomFloat(-m_dendriteAwayTowardMoveAmount, m_dendriteAwayTowardMoveAmount);
					dendrite.m_Y += getRandomFloat(-m_dendriteAwayTowardMoveAmount, m_dendriteAwayTowardMoveAmount);
					dendrite.m_Z += getRandomFloat(-m_dendriteAwayTowardMoveAmount, m_dendriteAwayTowardMoveAmount);
				}
				++cgpIdx;
				break;

			// If this value is lower than a set value or higher than a different
			// set value, the dendrite moves a static amount. Otherwise, no movement:
			case CGP::CGP_OUTPUT::NEAREST_AXON_CLOSER_FURTHER:
				if (cgpOutputs[cgpIdx] < m_dendriteLowMoveAway)
				{
					dendrite.m_X -= (dendrite.m_nearestAxonX - dendrite.m_X) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Y -= (dendrite.m_nearestAxonY - dendrite.m_Y) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Z -= (dendrite.m_nearestAxonZ - dendrite.m_Z) * m_dendriteAwayTowardMoveAmount;
				}
				else if (cgpOutputs[cgpIdx] > m_dendriteHighMoveToward)
				{
					dendrite.m_X += (dendrite.m_nearestAxonX - dendrite.m_X) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Y += (dendrite.m_nearestAxonY - dendrite.m_Y) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Z += (dendrite.m_nearestAxonZ - dendrite.m_Z) * m_dendriteAwayTowardMoveAmount;
				}
				// else: No movement.
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::STRONGEST_INPUT_CLOSER_FURTHER:
				if (cgpOutputs[cgpIdx] < m_dendriteLowMoveAway)
				{
					dendrite.m_X -= (dendrite.m_biggestInputX - dendrite.m_X) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Y -= (dendrite.m_biggestInputY - dendrite.m_Y) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Z -= (dendrite.m_biggestInputZ - dendrite.m_Z) * m_dendriteAwayTowardMoveAmount;
				}
				else if (cgpOutputs[cgpIdx] > m_dendriteHighMoveToward)
				{
					dendrite.m_X += (dendrite.m_biggestInputX - dendrite.m_X) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Y += (dendrite.m_biggestInputY - dendrite.m_Y) * m_dendriteAwayTowardMoveAmount;
					dendrite.m_Z += (dendrite.m_biggestInputZ - dendrite.m_Z) * m_dendriteAwayTowardMoveAmount;
				}
				// else: No movement.
				++cgpIdx;
				break;

			// Dendrite health not yet implemented.
			case CGP::CGP_OUTPUT::HEALTH:
					std::cout << "Dendrite health in CGP output params. Not yet implemented." << std::endl;
					++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::WEIGHT:
				dendrite.m_weight += static_cast<float>(cgpOutputs[cgpIdx]);
				dendrite.m_weight = fminf(fmaxf(dendrite.m_weight, m_dendriteMinWeight),
					m_dendriteMaxWeight);
				++cgpIdx;
				break;

			case CGP::CGP_OUTPUT::WEIGHT_HIGHER_LOWER:
				if (cgpOutputs[cgpIdx] < m_dendriteLowWeightDecrease)
					dendrite.m_weight -= m_dendriteWeightChangeAmount;
				else if (cgpOutputs[cgpIdx] > m_dendriteHighWeightIncrease)
					dendrite.m_weight += m_dendriteWeightChangeAmount;

				// Keep it in bounds:
				dendrite.m_weight = fminf(fmaxf(dendrite.m_weight, m_dendriteMinWeight),
					m_dendriteMaxWeight);
				++cgpIdx;
				break;
			}
		}
		// Make sure that after we move, we are still in a valid location:
		dendrite.constrainLocation(0.0, 0.0, 0.0, m_brainXSize, m_brainYSize, m_brainZSize);
		dendrite.constrainLength(parentNeuron.m_X, parentNeuron.m_Y, parentNeuron.m_Z,
			m_dendriteMaxLength);
	}

	std::vector<double> JBrain::getCGPInputs(const JDendrite& dendrite,
		const unsigned int& parentNeuronNum, const bool& outputNeuron)
	{
		std::vector<double> retVal;
		double tmpVal;
		JNeuron parentNeuron = outputNeuron ? m_outputNeurons[parentNeuronNum] :
			m_neurons[parentNeuronNum];

		// Every input must be special-cased:
		for (unsigned int i = 0; i < m_dendriteInputs.size(); ++i)
		{
			switch (m_dendriteInputs[i])
			{
			case CGP::CGP_INPUT::SAGE_MATCH_PERCENT:
				retVal.push_back(m_sageMatchPercent);
				break;

			case CGP::CGP_INPUT::CURRENT_WEIGHT:
				retVal.push_back(dendrite.m_weight);
				break;

			case CGP::CGP_INPUT::STRONGEST_INPUT_XYZ:
				retVal.push_back(dendrite.m_biggestInputX);
				retVal.push_back(dendrite.m_biggestInputY);
				retVal.push_back(dendrite.m_biggestInputZ);
				break;

			case CGP::CGP_INPUT::STRONGEST_INPUT_DISTANCE:
				retVal.push_back(dendrite.m_biggestInputDistance);
				break;

			case CGP::CGP_INPUT::STRONGEST_INPUT_IS_OBSERVATION_AXON:
				// Convert bool to double:
				tmpVal = 1.0;
				if (!dendrite.m_biggestInputIsEnvironmentAxon)
					tmpVal = 0.0;
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::STRONGEST_INPUT_VALUE:
				retVal.push_back(dendrite.m_biggestInputValue);
				break;

			case CGP::CGP_INPUT::NEAREST_AXON_XYZ:
				retVal.push_back(dendrite.m_nearestAxonX);
				retVal.push_back(dendrite.m_nearestAxonY);
				retVal.push_back(dendrite.m_nearestAxonZ);
				break;

			case CGP::CGP_INPUT::NEAREST_AXON_DISTANCE:
				retVal.push_back(dendrite.m_nearestAxonDistance);
				break;

			case CGP::CGP_INPUT::NEAREST_AXON_IS_OBSERVATION_AXON:
				// Convert bool to double:
				tmpVal = 1.0;
				if (!dendrite.m_nearestAxonIsEnvironmentAxon)
					tmpVal = 0.0;
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::NEAREST_AXON_IS_PART_OF_SAME_NEURON:
				// Convert bool to double:
				tmpVal = 1.0;
				if (!dendrite.m_nearestAxonPartOfSameNeuron)
					tmpVal = 0.0;
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::INPUT_MAGNITUDE:
				retVal.push_back(dendrite.m_currentValue);
				break;

			case CGP::CGP_INPUT::CURRENT_LENGTH:
				retVal.push_back(getDistance(dendrite, parentNeuron));
				break;

			case CGP::CGP_INPUT::NEURON_AGE:
				// Age as an input is always between 0.0 and 1.0. We divide the
				// age of the neuron by the maximum age we track.
				tmpVal = static_cast<double>(parentNeuron.m_age) /
					static_cast<double>(m_maxNeuronAge);
				tmpVal = fmin(1.0, tmpVal);
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::NEURON_HEALTH:
				retVal.push_back(parentNeuron.m_health);
				break;

			// The expected value minus the real output value should be
			// positive if the value needs to go up, negative if it needs to
			// go down:
			case CGP::CGP_INPUT::EXPECTED_OUTPUT_DIFF:
				// This assumes binary possible outputs. A lot needs to change
				// when that assumption changes.
				// Assume sage choice of 0.0. Set to 1.0 if it was the sage choice:
				float mostRecentSage = 0.0;
				if (m_sageChoices[parentNeuronNum] >= 0.5)
					mostRecentSage = 1.0;

				// Sage is expected, our brain output is actual:
				retVal.push_back(mostRecentSage - m_mostRecentBrainOutput[parentNeuronNum]);
				break;
			}
		}

		return retVal;
	}

	std::vector<double> JBrain::getCGPInputs(const JAxon& axon, const JNeuron& parentNeuron)
	{
		std::vector<double> retVal;
		double tmpVal;

		// All inputs require special processing:
		for (unsigned int i = 0; i < m_axonInputs.size(); ++i)
		{
			switch (m_axonInputs[i])
			{
			case CGP::CGP_INPUT::SAGE_MATCH_PERCENT:
				retVal.push_back(m_sageMatchPercent);
				break;

			case CGP::CGP_INPUT::NEAREST_DENDRITE_XYZ:
				retVal.push_back(axon.m_nearestDendriteX);
				retVal.push_back(axon.m_nearestDendriteY);
				retVal.push_back(axon.m_nearestDendriteZ);
				break;

			case CGP::CGP_INPUT::DENDRITE_TYPE:
				// 0 if attached to neuron, 1 if output dendrite:
				tmpVal = 0.0;
				if (axon.m_nearestDendriteIsActionDendrite)
					tmpVal = 1.0;
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::PERCENTAGE_FIRE:
				retVal.push_back(parentNeuron.getPercentageFire());
				break;

			case CGP::CGP_INPUT::PERCENTAGE_BRAIN_FIRE:
				retVal.push_back(m_averageNeuronFirePercentage);
				break;

			case CGP::CGP_INPUT::NEURON_AGE:
				// Age as an input is always between 0.0 and 1.0. We divide the
				// age of the neuron by the maximum age we track.
				tmpVal = static_cast<double>(parentNeuron.m_age) /
					static_cast<double>(m_maxNeuronAge);
				tmpVal = fmin(1.0, tmpVal);
				retVal.push_back(tmpVal);
				break;

			case CGP::CGP_INPUT::NEURON_HEALTH:
				retVal.push_back(parentNeuron.m_health);
				break;

			case CGP::CGP_INPUT::CURRENT_LENGTH:
				retVal.push_back(getDistance(axon, parentNeuron));
				break;
			}
		}

		return retVal;
	}

	std::vector<double> JBrain::getCGPInputs(const JNeuron& neuron)
	{
		std::vector<double> retVal;
		double age;

		// Each input must be special-cased:
		for (unsigned int i = 0; i < m_neuronInputs.size(); ++i)
		{
			switch (m_neuronInputs[i])
			{
			case CGP::CGP_INPUT::SAGE_MATCH_PERCENT:
				retVal.push_back(m_sageMatchPercent);
				break;

			case CGP::CGP_INPUT::NEURON_AGE:
				// Assume max age, change if needed:
				age = 1.0;
				if (neuron.m_age < m_maxNeuronAge)
					age = static_cast<double>(neuron.m_age) / static_cast<double>(m_maxNeuronAge);
				retVal.push_back(age);
				break;

			case CGP::CGP_INPUT::NEURON_HEALTH:		
				retVal.push_back(neuron.m_health);
				break;

			case CGP::CGP_INPUT::PERCENTAGE_FIRE:
				retVal.push_back(neuron.getPercentageFire());
				break;

			case CGP::CGP_INPUT::PERCENTAGE_BRAIN_FIRE:
				retVal.push_back(static_cast<double>(m_averageNeuronFirePercentage));
				break;
			}
		}

		return retVal;
	}

	void JBrain::applyAllCGP()
	{
		for (unsigned int i = 0; i < m_neurons.size(); ++i)
		{
			// Let the neuron apply its CGP program:
			applyCGP(m_neurons[i]);

			// Apply all of its dendrite's programs:
			for (unsigned int j = 0; j < m_neurons[i].m_dendrites.size(); ++j)
				applyCGP(m_neurons[i].m_dendrites[j], i, false);

			// Apply all of its axon's programs:
			for (unsigned int j = 0; j < m_neurons[i].m_axons.size(); ++j)
				applyCGP(m_neurons[i].m_axons[j], m_neurons[i]);
		}

		// Apply the output neuron program if needed:
		for (unsigned int i = 0; i < m_outputNeurons.size(); ++i)
			for (unsigned int j = 0; j < m_outputNeurons[i].m_dendrites.size(); ++j)
				applyCGP(m_outputNeurons[i].m_dendrites[j], i, true);
	}

	std::vector<float> JBrain::getValidCoordinatesWithinDistance(
		const float& startX, const float& startY, const float& startZ,
		const float& distance)
	{
		// Every value can be up to distance from the start:
		float minX = std::max(float(0.0), startX - distance);
		float maxX = std::min(m_brainXSize, startX + distance);
		float minY = std::max(float(0.0), startY - distance);
		float maxY = std::min(m_brainXSize, startY + distance);
		float minZ = std::max(float(0.0), startZ - distance);
		float maxZ = std::min(m_brainXSize, startZ + distance);

		// Choose randomly:
		float retX = getRandomFloat(minX, maxX);
		float retY = getRandomFloat(minY, maxY);
		float retZ = getRandomFloat(minZ, maxZ);
		float currDistance = getDistance(startX, startY, startZ, retX, retY, retZ);

		// Keep choosing until we get one within range:
		int attempts = 0;
		while (currDistance > distance)
		{	
			if (++attempts > 10)
			{
				// Narrow things until we find success:
				minX = std::min(startX, minX + float(0.5));
				maxX = std::max(startX, maxX - float(0.05));
				minY = std::min(startY, minY + float(0.5));
				maxY = std::max(startY, maxY - float(0.05));
				minZ = std::min(startZ, minZ + float(0.5));
				maxZ = std::max(startZ, maxZ - float(0.05));				
				// std::cout << currDistance << " > " << distance << std::endl;
			}

			retX = getRandomFloat(minX, maxX);
			retY = getRandomFloat(minY, maxY);
			retZ = getRandomFloat(minZ, maxZ);
			currDistance = getDistance(startX, startY, startZ, retX, retY, retZ);
		}

		return std::vector<float> {retX, retY, retZ};
	}

	void JBrain::addRandomAxon(JNeuron& neuron)
	{
		// Get coordinates within a valid distance of the neuron:
		auto coords = getValidCoordinatesWithinDistance(
			neuron.m_X, neuron.m_Y, neuron.m_Z, m_axonMaxLength);

		// Create and add the dendrite to the neuron's vector:
		neuron.m_axons.push_back(
			JAxon(coords[0], coords[1], coords[2]));
	}

	void JBrain::addRandomDendrite(JNeuron& neuron)
	{
		// Get coordinates within a valid distance of the neuron:
		auto coords = getValidCoordinatesWithinDistance(
			neuron.m_X, neuron.m_Y, neuron.m_Z, m_dendriteMaxLength);

		// Create and add the dendrite to the neuron's vector:
		neuron.m_dendrites.push_back(
			JDendrite(coords[0], coords[1], coords[2],
			  getRandomFloat(m_dendriteMinWeight, m_dendriteMaxWeight)));
	}
	
	JNeuron JBrain::createNewNeuron(const float& x, const float& y, const float& z,
		const int& dendriteCount, const int& axonCount)
	{
		JNeuron neuron(
			x, y, z,
			getRandomFloat(m_neuronMinFireValue, m_neuronMaxFireValue),
			m_neuronFireThreshold, m_neuronStartingHealth,
			getNextNeuronNumber());

		// Determine how many axon and dendrites to add. -1 indicates random:
		int totalDendrites = dendriteCount;
		int totalAxon = axonCount;
		
		if (totalDendrites == -1)
		{
			totalDendrites = getRandomInt(static_cast<int>(m_dendriteMinCount),
				static_cast<int>(m_dendriteMaxCount));
		}
		if (totalAxon == -1)
		{
			totalAxon = getRandomInt(static_cast<int>(m_axonMinCount),
				static_cast<int>(m_axonMaxCount));
		}

		// Add the dendrites and axon:
		for (int i = 0; i < totalDendrites; ++i)
			addRandomDendrite(neuron);

		for (int i = 0; i < totalAxon; ++i)
			addRandomAxon(neuron);

		return neuron;
	}

	void JBrain::addRandomStartingNeurons(bool destroyCurrentNeurons)
	{
		// If we are resetting a brain to fresh neurons, do that here:
		if (destroyCurrentNeurons)
		{
			m_neurons.clear();
			m_neuronFires.clear();
		}

		// Starting neuron count is a random choice minus how many
		// we already have. For totally random brains, this is all of
		// their neurons. Descendent brains may have duplicates of
		// some of their parent neurons.
		unsigned int neuronsNeeded = static_cast<unsigned int>(
			getRandomInt(m_minStartingNeurons, m_maxStartingNeurons))
			- static_cast<int>(m_neurons.size());

		for (unsigned int i = 0; i < neuronsNeeded; ++i)
		{
			// Create it at random coordinates. We assume 0, 0, 0 is one corner
			// of the brain and all coordinates are positive. Thus, a brain with size
			// X, Y, Z has available space of 0-X, 0-Y, 0-Z:
			m_neurons.push_back(createNewNeuron(
				getRandomFloat(0.0, m_brainXSize),
				getRandomFloat(0.0, m_brainYSize),
				getRandomFloat(0.0, m_brainZSize)));			
		}
	}

	bool JBrain::areEqual(CGP::JBrainCGPIndividual* lhs, CGP::JBrainCGPIndividual* rhs)
	{		
		// Check for both null or pointing to the same object:
		if (lhs == rhs)
			return true;

		// Can't both be null; if either is, return false:
		if (lhs == nullptr || rhs == nullptr)
			return false;

		// Both not null, not pointing at the same object. Return the
		// result of their equality operator:
		return (*lhs == *rhs);
	}

	bool JBrain::operator==(const JBrain& rhs)
	{
		if (this == &rhs)
			return true;

		bool retVal = true;

		// A lot of variables to compare:
		return areEqual(m_CGPDendriteUpdater, rhs.m_CGPDendriteUpdater) &&
			areEqual(m_CGPOutputDendriteUpdater, rhs.m_CGPOutputDendriteUpdater) &&
			(m_name == rhs.m_name) &&
			(m_parentName == rhs.m_parentName) &&
			(m_observationSize == rhs.m_observationSize) &&
			(m_actionSize == rhs.m_actionSize) &&
			(m_outputDendrites == rhs.m_outputDendrites) &&
			(m_inputAxons == rhs.m_inputAxons) &&
			(fabs(m_dendriteMaxLength - rhs.m_dendriteMaxLength) < FLT_EPSILON) &&
			(m_dendriteMinCount == rhs.m_dendriteMinCount) &&
			(m_dendriteMaxCount == rhs.m_dendriteMaxCount) &&
			(fabs(m_dendriteMinWeight - rhs.m_dendriteMinWeight) < FLT_EPSILON) &&
			(fabs(m_dendriteMaxWeight - rhs.m_dendriteMaxWeight) < FLT_EPSILON) &&
			(fabs(m_dendriteLowMoveAway - rhs.m_dendriteLowMoveAway) < FLT_EPSILON) &&
			(fabs(m_dendriteHighMoveToward - rhs.m_dendriteHighMoveToward) < FLT_EPSILON) &&
			(fabs(m_dendriteAwayTowardMoveAmount - rhs.m_dendriteAwayTowardMoveAmount) < FLT_EPSILON) &&
			(fabs(m_dendriteLowWeightDecrease - rhs.m_dendriteLowWeightDecrease) < FLT_EPSILON) &&
			(fabs(m_dendriteHighWeightIncrease - rhs.m_dendriteHighWeightIncrease) < FLT_EPSILON) &&
			(fabs(m_dendriteWeightChangeAmount - rhs.m_dendriteWeightChangeAmount) < FLT_EPSILON) &&
			(fabs(m_axonMaxLength - rhs.m_axonMaxLength) < FLT_EPSILON) &&
			(m_axonMinCount == rhs.m_axonMinCount) &&
			(m_axonMaxCount == rhs.m_axonMaxCount) &&
			(fabs(m_axonLowMoveAway - rhs.m_axonLowMoveAway) < FLT_EPSILON) &&
			(fabs(m_axonHighMoveToward - rhs.m_axonHighMoveToward) < FLT_EPSILON) &&
			(fabs(m_axonAwayTowardMoveAmount - rhs.m_axonAwayTowardMoveAmount) < FLT_EPSILON) &&
			(m_neuronProbabilisticFire == rhs.m_neuronProbabilisticFire) &&
			(fabs(m_neuronFireThreshold - rhs.m_neuronFireThreshold) < FLT_EPSILON) &&
			(fabs(m_neuronMinFireValue - rhs.m_neuronMinFireValue) < FLT_EPSILON) &&
			(fabs(m_neuronMaxFireValue - rhs.m_neuronMaxFireValue) < FLT_EPSILON) &&
			(m_neuronUseDynamicFireThresholds == rhs.m_neuronUseDynamicFireThresholds) &&
			(fabs(m_neuronFireThresholdIdleChange - rhs.m_neuronFireThresholdIdleChange) < FLT_EPSILON) &&
			(fabs(m_neuronFireThresholdActiveChange - rhs.m_neuronFireThresholdActiveChange) < FLT_EPSILON) &&
			(m_neuronRefractoryPeriod == rhs.m_neuronRefractoryPeriod) &&
			(m_neuronDuplicateNearby == rhs.m_neuronDuplicateNearby) &&
			(fabs(m_neuronMinNearbyDistance - rhs.m_neuronMinNearbyDistance) < FLT_EPSILON) &&
			(fabs(m_neuronMaxNearbyDistance - rhs.m_neuronMaxNearbyDistance) < FLT_EPSILON) &&
			(m_minStartingNeurons == rhs.m_minStartingNeurons) &&
			(m_maxStartingNeurons == rhs.m_maxStartingNeurons) &&
			(m_maxNeurons == rhs.m_maxNeurons) &&
			(m_useOutputNeurons == rhs.m_useOutputNeurons) &&
			(fabs(m_neuronDeathHealth - rhs.m_neuronDeathHealth) < FLT_EPSILON) &&
			(fabs(m_neuronDeathHealth_Original - rhs.m_neuronDeathHealth_Original) < FLT_EPSILON) &&
			(fabs(m_neuronDuplicateHealth - rhs.m_neuronDuplicateHealth) < FLT_EPSILON) &&
			(fabs(m_neuronDuplicateHealth_Original - rhs.m_neuronDuplicateHealth_Original) < FLT_EPSILON) &&
			(fabs(m_neuronDeathDuplicateHealthThresholdMultiplier - rhs.m_neuronDeathDuplicateHealthThresholdMultiplier) < FLT_EPSILON) &&
			(fabs(m_neuronDuplicationHealthChange - rhs.m_neuronDuplicationHealthChange) < FLT_EPSILON) &&
			(m_neuronDuplicationHealthReset == rhs.m_neuronDuplicationHealthReset) &&
			(m_jNeuronActivationFunction == rhs.m_jNeuronActivationFunction) &&
			(m_neuronFires == rhs.m_neuronFires) &&
			(fabs(m_neuronFireSpaceDeterioration - rhs.m_neuronFireSpaceDeterioration) < FLT_EPSILON) &&
			(fabs(m_neuronFireTimeDeterioration - rhs.m_neuronFireTimeDeterioration) < FLT_EPSILON) &&
			(m_neuronFireLifetime == rhs.m_neuronFireLifetime) &&
			(m_inputNeuronFiresAge == rhs.m_inputNeuronFiresAge) &&
			(m_currNeuronNumber == rhs.m_currNeuronNumber) &&
			(m_usePreTrainSleep == rhs.m_usePreTrainSleep) &&
			(m_usePostTrainSleep == rhs.m_usePostTrainSleep) &&
			(fabs(m_brainXSize - rhs.m_brainXSize) < FLT_EPSILON) &&
			(fabs(m_brainYSize - rhs.m_brainYSize) < FLT_EPSILON) &&
			(fabs(m_brainZSize - rhs.m_brainZSize) < FLT_EPSILON) &&
			(m_brainUseSameDimensions == rhs.m_brainUseSameDimensions) &&
			(m_brainResetBeforeProcessingInput == rhs.m_brainResetBeforeProcessingInput) &&
			(m_brainInputsOnOneSide == rhs.m_brainInputsOnOneSide) &&
			(m_brainOutputsOnOneSide == rhs.m_brainOutputsOnOneSide) &&
			(m_brainOutputsIgnoreEnvironmentInputs == rhs.m_brainOutputsIgnoreEnvironmentInputs) &&
			(m_brainProcessingStepsBetweenInputAndOutput == rhs.m_brainProcessingStepsBetweenInputAndOutput) &&
			(m_brainOutputsToAverageTogether == rhs.m_brainOutputsToAverageTogether) &&
			(fabs(m_minP - rhs.m_minP) < FLT_EPSILON) &&
			(fabs(m_maxP - rhs.m_maxP) < FLT_EPSILON) &&
			(fabs(m_minConstraint - rhs.m_minConstraint) < FLT_EPSILON) &&
			(fabs(m_maxConstraint - rhs.m_maxConstraint) < FLT_EPSILON) &&
			(fabs(m_averageNeuronFirePercentage - rhs.m_averageNeuronFirePercentage) < FLT_EPSILON) &&
			(fabs(m_sageMatchPercent - rhs.m_sageMatchPercent) < FLT_EPSILON) &&
			(m_maxNeuronAge == rhs.m_maxNeuronAge) &&
			(fabs(m_neuronStartingHealth - rhs.m_neuronStartingHealth) < FLT_EPSILON) &&
			(fabs(m_neuronCGPOutputLowHealthChange - rhs.m_neuronCGPOutputLowHealthChange) < FLT_EPSILON) &&
			(fabs(m_neuronCGPOutputHighHealthChange - rhs.m_neuronCGPOutputHighHealthChange) < FLT_EPSILON) &&
			(fabs(m_neuronCGPOutputHealthChangeAmount - rhs.m_neuronCGPOutputHealthChangeAmount) < FLT_EPSILON) &&
			//areEqual(m_CGPDendriteUpdater, rhs.m_CGPDendriteUpdater) &&
			areEqual(m_CGPAxonUpdater, rhs.m_CGPAxonUpdater) &&
			areEqual(m_CGPNeuronUpdater, rhs.m_CGPNeuronUpdater) &&
			areEqual(m_CGPChemicalUpdater, rhs.m_CGPChemicalUpdater) &&
			(m_dendriteInputs == rhs.m_dendriteInputs) &&
			(m_dendriteOutputs == rhs.m_dendriteOutputs) &&
			(m_axonInputs == rhs.m_axonInputs) &&
			(m_axonOutputs == rhs.m_axonOutputs) &&
			(m_neuronInputs == rhs.m_neuronInputs) &&
			(m_neuronOutputs == rhs.m_neuronOutputs) &&
			(m_dendriteProgramNodes == rhs.m_dendriteProgramNodes) &&
			(m_axonProgramNodes == rhs.m_axonProgramNodes) &&
			(m_neuronProgramNodes == rhs.m_neuronProgramNodes) &&
			(m_updateEvent == rhs.m_updateEvent) &&
			(m_updateFrequency == rhs.m_updateFrequency) &&
			(m_functionStringList == rhs.m_functionStringList) &&
			(m_neurons == rhs.m_neurons) &&
			(m_sageChoices == rhs.m_sageChoices) &&
			(m_brainChoices == rhs.m_brainChoices) &&
			(m_inputProcessingsSinceLastUpdate == rhs.m_inputProcessingsSinceLastUpdate);
	}

	JBrain::JBrain(std::string yamlFilename)
	{
		YAML::Node fullConfig = YAML::LoadFile(yamlFilename);

		// All values should be written directly out in a single brain section:
		m_name = fullConfig["name"].as<std::string>();
		m_dendriteMaxLength = fullConfig["dendriteMaxLength"].as<float>();
		m_dendriteMinCount = fullConfig["dendriteMinCount"].as<unsigned int>();
		m_dendriteMaxCount = fullConfig["dendriteMaxCount"].as<unsigned int>();
		m_axonMaxLength = fullConfig["axonMaxLength"].as<float>();
		m_axonMinCount = fullConfig["axonMinCount"].as<unsigned int>();
		m_axonMaxCount = fullConfig["axonMaxCount"].as<unsigned int>();
		m_neuronProbabilisticFire = fullConfig["neuronProbabilisticFire"].as<bool>();
		m_neuronFireThreshold = fullConfig["neuronFireThreshold"].as<float>();
		m_neuronRefractoryPeriod = fullConfig["neuronRefractoryPeriod"].as<unsigned int>();
		m_neuronDuplicateNearby = fullConfig["neuronDuplicateNearby"].as<bool>();
		m_neuronMinNearbyDistance = fullConfig["neuronMinNearbyDistance"].as<float>();
		m_neuronMaxNearbyDistance = fullConfig["neuronMaxNearbyDistance"].as<float>();
		m_usePreTrainSleep = fullConfig["usePreTrainSleep"].as<bool>();
		m_usePostTrainSleep = fullConfig["usePostTrainSleep"].as<bool>();
		m_brainXSize = fullConfig["brainXSize"].as<float>();
		m_brainYSize = fullConfig["brainYSize"].as<float>();
		m_brainZSize = fullConfig["brainZSize"].as<float>();
		m_brainUseSameDimensions = fullConfig["brainUseSameDimensions"].as<bool>();
		
		// Let the update functions read themselves in:
		m_CGPDendriteUpdater = nullptr;
		m_CGPAxonUpdater = nullptr;
		m_CGPNeuronUpdater = nullptr;
		m_CGPChemicalUpdater = nullptr;

		if (fullConfig["CGPDendriteUpdater"])
		{
			// m_CGPDendriteUpdater = new CGP::JBrainCGPIndividual(fullConfig["CGPDendriteUpdater"])
		}
		
		if (fullConfig["CGPAxonUpdater"])
		{
			// m_CGPAxonUpdater = new CGP::JBrainCGPIndividual(fullConfig["CGPAxonUpdater"])
		}

		if (fullConfig["CGPNeuronUpdater"])
		{
			// m_CGPNeuronUpdater = new CGP::JBrainCGPIndividual(fullConfig["CGPNeuronUpdater"])
		}

		if (fullConfig["CGPChemicalUpdater"])
		{
			// m_CGPChemicalUpdater = new CGP::JBrainCGPIndividual(fullConfig["CGPChemicalUpdater"])
		}
	}

	void JBrain::writeSelfToJson(json& j)
	{
		json dendCGP;
		json outDendCGP;
		json axonCGP;
		json neuronCGP;
		json chemCGP;

		// Ask each updater to write itself to a json node:
		if (m_CGPDendriteUpdater != nullptr)
			m_CGPDendriteUpdater->writeSelfToJson(dendCGP);
		else
			dendCGP = "NULL";

		if (m_CGPOutputDendriteUpdater != nullptr)
			m_CGPOutputDendriteUpdater->writeSelfToJson(outDendCGP);
		else
			outDendCGP = "NULL";

		if (m_CGPAxonUpdater != nullptr)
			m_CGPAxonUpdater->writeSelfToJson(axonCGP);
		else
			axonCGP = "NULL";

		if (m_CGPNeuronUpdater != nullptr)
			m_CGPNeuronUpdater->writeSelfToJson(neuronCGP);
		else
			neuronCGP = "NULL";

		if (m_CGPChemicalUpdater != nullptr)
			m_CGPChemicalUpdater->writeSelfToJson(chemCGP);
		else
			chemCGP = "NULL";

		j["CGPDendriteUpdater"] = dendCGP;
		j["CGPOutputDendriteUpdater"] = outDendCGP;
		j["CGPAxonUpdater"] = axonCGP;
		j["CGPNeuronUpdater"] = neuronCGP;
		j["CGPChemicalUpdater"] = chemCGP;

		// Need the rest of the brain variables
		j["name"] = m_name;
		j["parentName"] = m_parentName;
		j["observationSize"] = m_observationSize;
		j["actionSize"] = m_actionSize;
		j["dendriteMaxLength"] = m_dendriteMaxLength;
		j["dendriteMinCount"] = m_dendriteMinCount;
		j["dendriteMaxCount"] = m_dendriteMaxCount;
		j["dendriteMinWeight"] = m_dendriteMinWeight;
		j["dendriteMaxWeight"] = m_dendriteMaxWeight;
		j["dendriteLowMoveAway"] = m_dendriteLowMoveAway;
		j["dendriteHighMoveToward"] = m_dendriteHighMoveToward;
		j["dendriteAwayTowardMoveAmount"] = m_dendriteAwayTowardMoveAmount;
		j["dendriteLowWeightDecrease"] = m_dendriteLowWeightDecrease;
		j["dendriteHighWeightIncrease"] = m_dendriteHighWeightIncrease;
		j["dendriteWeightChangeAmount"] = m_dendriteWeightChangeAmount;
		j["axonMaxLength"] = m_axonMaxLength;
		j["axonMinCount"] = m_axonMinCount;
		j["axonMaxCount"] = m_axonMaxCount;
		j["axonLowMoveAway"] = m_axonLowMoveAway;
		j["axonHighMoveToward"] = m_axonHighMoveToward;
		j["axonAwayTowardMoveAmount"] = m_axonAwayTowardMoveAmount;
		j["neuronProbabilisticFire"] = m_neuronProbabilisticFire;
		j["neuronFireThreshold"] = m_neuronFireThreshold;
		j["neuronMinFireValue"] = m_neuronMinFireValue;
		j["neuronMaxFireValue"] = m_neuronMaxFireValue;
		j["neuronUseDynamicFireThresholds"] = m_neuronUseDynamicFireThresholds;
		j["neuronFireThresholdIdleChange"] = m_neuronFireThresholdIdleChange;
		j["neuronFireThresholdActiveChange"] = m_neuronFireThresholdActiveChange;
		j["neuronRefractoryPeriod"] = m_neuronRefractoryPeriod;
		j["neuronDuplicateNearby"] = m_neuronDuplicateNearby;
		j["neuronMinNearbyDistance"] = m_neuronMinNearbyDistance;
		j["neuronMaxNearbyDistance"] = m_neuronMaxNearbyDistance;
		j["minStartingNeurons"] = m_minStartingNeurons;
		j["maxStartingNeurons"] = m_maxStartingNeurons;
		j["maxNeurons"] = m_maxNeurons;
		j["useOutputNeurons"] = m_useOutputNeurons;
		j["neuronStartingHealth"] = m_neuronStartingHealth;
		j["neuronCGPOutputLowHealthChange"] = m_neuronCGPOutputLowHealthChange;
		j["neuronCGPOutputHighHealthChange"] = m_neuronCGPOutputHighHealthChange;
		j["neuronCGPOutputHealthChangeAmount"] = m_neuronCGPOutputHealthChangeAmount;
		j["neuronDeathHealth"] = m_neuronDeathHealth;
		j["neuronDeathHealth_Original"] = m_neuronDeathHealth_Original;
		j["neuronDuplicateHealth"] = m_neuronDuplicateHealth;
		j["neuronDuplicateHealth_Original"] = m_neuronDuplicateHealth_Original;
		j["neuronDeathDuplicateHealthThresholdMultiplier"] = m_neuronDeathDuplicateHealthThresholdMultiplier;
		j["neuronDuplicationHealthChange"] = m_neuronDuplicationHealthChange;
		j["neuronDuplicationHealthReset"] = m_neuronDuplicationHealthReset;
		j["jNeuronActivationFunction"] = CGP::ActivationFunctionToString(m_jNeuronActivationFunction);
		j["neuronFireSpaceDeterioration"] = m_neuronFireSpaceDeterioration;
		j["neuronFireTimeDeterioration"] = m_neuronFireTimeDeterioration;
		j["neuronFireLifetime"] = m_neuronFireLifetime;
		j["inputNeuronFiresAge"] = m_inputNeuronFiresAge;
		j["currNeuronNumber"] = m_currNeuronNumber;
		j["usePreTrainSleep"] = m_usePreTrainSleep;
		j["usePostTrainSleep"] = m_usePostTrainSleep;
		j["brainXSize"] = m_brainXSize;
		j["brainYSize"] = m_brainYSize;
		j["brainZSize"] = m_brainZSize;
		j["brainUseSameDimensions"] = m_brainUseSameDimensions;
		j["brainResetBeforeProcessingInput"] = m_brainResetBeforeProcessingInput;
		j["brainProcessingStepsBetweenInputAndOutput"] = m_brainProcessingStepsBetweenInputAndOutput;
		j["brainOutputsToAverageTogether"] = m_brainOutputsToAverageTogether;
		j["brainInputsOnOneSide"] = m_brainInputsOnOneSide;
		j["brainOutputsOnOneSide"] = m_brainOutputsOnOneSide;
		j["brainOutputsIgnoreEnvironmentInputs"] = m_brainOutputsIgnoreEnvironmentInputs;
		j["minP"] = m_minP;
		j["maxP"] = m_maxP;
		j["minConstraint"] = m_minConstraint;
		j["maxConstraint"] = m_maxConstraint;
		j["maxNeuronAge"] = m_maxNeuronAge;		 
		j["dendriteProgramNodes"] = m_dendriteProgramNodes;
		j["outputDendriteProgramNodes"] = m_outputDendriteProgramNodes;
		j["axonProgramNodes"] = m_axonProgramNodes;		 
		j["neuronProgramNodes"] = m_neuronProgramNodes;		 
		j["updateFrequency"] = m_updateFrequency;
		j["updateEvent"] = CGP::UpdateEventToString(m_updateEvent);

		// Write out our outputDendrites and inputAxons:
		// Dendrites and axon have many other properties, but they are
		// updated with each brain time step. These values will be invalid
		// with the freshly-read-in brain:
		j["outputDendrites"] = json::array();
		for (unsigned int i = 0; i < m_outputDendrites.size(); ++i)
		{
			j["outputDendrites"][i]["X"] = m_outputDendrites[i].m_X;
			j["outputDendrites"][i]["Y"] = m_outputDendrites[i].m_Y;
			j["outputDendrites"][i]["Z"] = m_outputDendrites[i].m_Z;
			j["outputDendrites"][i]["weight"] = m_outputDendrites[i].m_weight;			
		}

		j["outputNeurons"] = json::array();
		for (unsigned int n = 0; n < m_outputNeurons.size(); ++n)
		{
			j["outputNeurons"][n]["X"] = m_outputNeurons[n].m_X;
			j["outputNeurons"][n]["Y"] = m_outputNeurons[n].m_Y;
			j["outputNeurons"][n]["Z"] = m_outputNeurons[n].m_Z;
			j["outputNeurons"][n]["neuronNumber"] = m_outputNeurons[n].m_neuronNumber;
			j["outputNeurons"][n]["age"] = m_outputNeurons[n].m_age;

			j["outputNeurons"][n]["dendrites"] = json::array();
			for (unsigned int d = 0; d < m_outputNeurons[n].m_dendrites.size(); ++d)
			{
				j["outputNeurons"][n]["dendrites"][d]["X"] = m_outputNeurons[n].m_dendrites[d].m_X;
				j["outputNeurons"][n]["dendrites"][d]["Y"] = m_outputNeurons[n].m_dendrites[d].m_Y;
				j["outputNeurons"][n]["dendrites"][d]["Z"] = m_outputNeurons[n].m_dendrites[d].m_Z;
				j["outputNeurons"][n]["dendrites"][d]["weight"] = m_outputNeurons[n].m_dendrites[d].m_weight;
			}
		}

		j["inputAxons"] = json::array();
		for (unsigned int i = 0; i < m_inputAxons.size(); ++i)
		{
			j["inputAxons"][i]["X"] = m_inputAxons[i].m_X;
			j["inputAxons"][i]["Y"] = m_inputAxons[i].m_Y;
			j["inputAxons"][i]["Z"] = m_inputAxons[i].m_Z;
		}

		j["neurons"] = json::array();
		for (unsigned int n = 0; n < m_neurons.size(); ++n)
		{
			j["neurons"][n]["X"] = m_neurons[n].m_X;
			j["neurons"][n]["Y"] = m_neurons[n].m_Y;
			j["neurons"][n]["Z"] = m_neurons[n].m_Z;
			j["neurons"][n]["fireValue"] = m_neurons[n].m_fireValue;
			j["neurons"][n]["fireThreshold"] = m_neurons[n].m_fireThreshold;
			j["neurons"][n]["health"] = m_neurons[n].m_health;
			j["neurons"][n]["fireOpportunitiesSinceLastUpdate"] = 
				m_neurons[n].m_fireOpportunitiesSinceLastUpdate;
			j["neurons"][n]["timesFiredSinceLastUpdate"] = 
				m_neurons[n].m_timesFiredSinceLastUpdate;
			j["neurons"][n]["neuronNumber"] = m_neurons[n].m_neuronNumber;
			j["neurons"][n]["timeStepsSinceLastFire"] =
				m_neurons[n].m_timeStepsSinceLastFire;
			j["neurons"][n]["age"] = m_neurons[n].m_age;
			
			j["neurons"][n]["axons"] = json::array();
			for (unsigned int a = 0; a < m_neurons[n].m_axons.size(); ++a)
			{
				j["neurons"][n]["axons"][a]["X"] = m_neurons[n].m_axons[a].m_X;
				j["neurons"][n]["axons"][a]["Y"] = m_neurons[n].m_axons[a].m_Y;
				j["neurons"][n]["axons"][a]["Z"] = m_neurons[n].m_axons[a].m_Z;
			}

			j["neurons"][n]["dendrites"] = json::array();
			for (unsigned int d = 0; d < m_neurons[n].m_dendrites.size(); ++d)
			{
				j["neurons"][n]["dendrites"][d]["X"] = m_neurons[n].m_dendrites[d].m_X;
				j["neurons"][n]["dendrites"][d]["Y"] = m_neurons[n].m_dendrites[d].m_Y;
				j["neurons"][n]["dendrites"][d]["Z"] = m_neurons[n].m_dendrites[d].m_Z;
				j["neurons"][n]["dendrites"][d]["weight"] = m_neurons[n].m_dendrites[d].m_weight;
			}
		}

		// Convert each vector of enums into equivalent strings:
		std::vector<std::string> neuronInputsStrings;
		std::vector<std::string> neuronOutputsStrings;
		std::vector<std::string> axonInputsStrings;
		std::vector<std::string> axonOutputsStrings;
		std::vector<std::string> dendriteInputsStrings;
		std::vector<std::string> dendriteOutputsStrings;
		std::vector<std::string> outputDendriteInputsStrings;
		std::vector<std::string> outputDendriteOutputsStrings;

		for (const auto& elem : m_neuronInputs)
			neuronInputsStrings.push_back(CGP::CGPInputToString(elem));
		j["neuronInputs"] = neuronInputsStrings;

		for (const auto& elem : m_neuronOutputs)
			neuronOutputsStrings.push_back(CGP::CGPOutputToString(elem));
		j["neuronOutputs"] = neuronOutputsStrings;

		for (const auto& elem : m_axonInputs)
			axonInputsStrings.push_back(CGP::CGPInputToString(elem));
		j["axonInputs"] = axonInputsStrings;

		for (const auto& elem : m_axonOutputs)
			axonOutputsStrings.push_back(CGP::CGPOutputToString(elem));
		j["axonOutputs"] = axonOutputsStrings;

		for (const auto& elem : m_dendriteInputs)
			dendriteInputsStrings.push_back(CGP::CGPInputToString(elem));
		j["dendriteInputs"] = dendriteInputsStrings;

		for (const auto& elem : m_dendriteOutputs)
			dendriteOutputsStrings.push_back(CGP::CGPOutputToString(elem));
		j["dendriteOutputs"] = dendriteOutputsStrings;

		for (const auto& elem : m_outputDendriteInputs)
			outputDendriteInputsStrings.push_back(CGP::CGPInputToString(elem));
		j["outputDendriteInputs"] = outputDendriteInputsStrings;

		for (const auto& elem : m_outputDendriteOutputs)
			outputDendriteOutputsStrings.push_back(CGP::CGPOutputToString(elem));
		j["outputDendriteOutputs"] = outputDendriteOutputsStrings;

		j["functionStringList"] = m_functionStringList;
		j["sageChoices"] = m_sageChoices;
		j["brainChoices"] = m_brainChoices;
	}

	JBrain* JBrain::getBrainFromJson(json& j)
	{
		// For each CGP Individual, if it isn't a string, it should be
		// a struct, load it:
		CGP::JBrainCGPIndividual* dendCGP = nullptr;
		CGP::JBrainCGPIndividual* outDendCGP = nullptr;
		CGP::JBrainCGPIndividual* axonCGP = nullptr;
		CGP::JBrainCGPIndividual* neuronCGP = nullptr;
		CGP::JBrainCGPIndividual* chemCGP = nullptr;

		if (!j["CGPDendriteUpdater"].is_string())
			dendCGP = CGP::JBrainCGPIndividual::getCGPIndividualFromJson(j["CGPDendriteUpdater"]);

		if (!j["CGPOutputDendriteUpdater"].is_string())
			outDendCGP = CGP::JBrainCGPIndividual::getCGPIndividualFromJson(j["CGPOutputDendriteUpdater"]);
		
		if (!j["CGPAxonUpdater"].is_string())
			axonCGP = CGP::JBrainCGPIndividual::getCGPIndividualFromJson(j["CGPAxonUpdater"]);

		if (!j["CGPNeuronUpdater"].is_string())
			neuronCGP = CGP::JBrainCGPIndividual::getCGPIndividualFromJson(j["CGPNeuronUpdater"]);

		if (!j["CGPChemicalUpdater"].is_string())
			chemCGP = CGP::JBrainCGPIndividual::getCGPIndividualFromJson(j["CGPChemicalUpdater"]);

		// Get the strings of inputs/functions:
		std::vector<std::string> neuronInputsStrings = j["neuronInputs"];
		std::vector<std::string> neuronOutputsStrings = j["neuronOutputs"];
		std::vector<std::string> axonInputsStrings = j["axonInputs"];
		std::vector<std::string> axonOutputsStrings = j["axonOutputs"];
		std::vector<std::string> dendriteInputsStrings = j["dendriteInputs"];
		std::vector<std::string> dendriteOutputsStrings = j["dendriteOutputs"];
		std::vector<std::string> outputDendriteInputsStrings = j["outputDendriteInputs"];
		std::vector<std::string> outputDendriteOutputsStrings = j["outputDendriteOutputs"];
		
		// Convert those strings back to the enums we need:
		std::vector<CGP::CGP_INPUT> neuronInputs;
		std::vector<CGP::CGP_OUTPUT> neuronOutputs;		
		std::vector<CGP::CGP_INPUT> axonInputs;
		std::vector<CGP::CGP_OUTPUT> axonOutputs;
		std::vector<CGP::CGP_INPUT> dendriteInputs;
		std::vector<CGP::CGP_OUTPUT> dendriteOutputs;
		std::vector<CGP::CGP_INPUT> outputDendriteInputs;
		std::vector<CGP::CGP_OUTPUT> outputDendriteOutputs;
		
		for (const auto& elem : neuronInputsStrings)
			neuronInputs.push_back(CGP::StringToCGPInput(elem));
		
		for (const auto& elem : neuronOutputsStrings)
			neuronOutputs.push_back(CGP::StringToCGPOutput(elem));
		
		for (const auto& elem : axonInputsStrings)
			axonInputs.push_back(CGP::StringToCGPInput(elem));

		for (const auto& elem : axonOutputsStrings)
			axonOutputs.push_back(CGP::StringToCGPOutput(elem));

		for (const auto& elem : dendriteInputsStrings)
			dendriteInputs.push_back(CGP::StringToCGPInput(elem));

		for (const auto& elem : dendriteOutputsStrings)
			dendriteOutputs.push_back(CGP::StringToCGPOutput(elem));

		for (const auto& elem : outputDendriteInputsStrings)
			outputDendriteInputs.push_back(CGP::StringToCGPInput(elem));

		for (const auto& elem : outputDendriteOutputsStrings)
			outputDendriteOutputs.push_back(CGP::StringToCGPOutput(elem));

		// Get/Convert the function list:
		std::vector<std::string> functionStringList = j["functionStringList"];
		std::vector<std::function<double(double, double, double)> > functionList;

		for (const auto& elem : functionStringList)
			functionList.push_back(CGPFunctions::doubleIn_doubleOut::getFuncFromString(elem));

		JBrain* retVal = new JBrain(
			j["name"].get<std::string>(),
			j["parentName"].get<std::string>(),
			j["observationSize"].get<unsigned int>(),
			j["actionSize"].get<unsigned int>(),
			j["dendriteMaxLength"].get<float>(),
			j["dendriteMinCount"].get<unsigned int>(),
			j["dendriteMaxCount"].get<unsigned int>(),
			j["dendriteMinWeight"].get<float>(),
			j["dendriteMaxWeight"].get<float>(),
			j["dendriteLowMoveAway"].get<float>(),
			j["dendriteHighMoveToward"].get<float>(),
			j["dendriteAwayTowardMoveAmount"].get<float>(),
			j["dendriteLowWeightDecrease"].get<float>(),
			j["dendriteHighWeightIncrease"].get<float>(),
			j["dendriteWeightChangeAmount"].get<float>(),
			j["axonMaxLength"].get<float>(),
			j["axonMinCount"].get<unsigned int>(),
			j["axonMaxCount"].get<unsigned int>(),
			j["axonLowMoveAway"].get<float>(),
			j["axonHighMoveToward"].get<float>(),
			j["axonAwayTowardMoveAmount"].get<float>(),
			j["neuronProbabilisticFire"].get<bool>(),
			j["neuronFireThreshold"].get<float>(),
			j["neuronMinFireValue"].get<float>(),
			j["neuronMaxFireValue"].get<float>(),
			j["neuronUseDynamicFireThresholds"].get<bool>(),
			j["neuronFireThresholdIdleChange"].get<float>(),
			j["neuronFireThresholdActiveChange"].get<float>(),
			j["neuronRefractoryPeriod"].get<unsigned int>(),
			j["neuronDuplicateNearby"].get<bool>(),
			j["neuronMinNearbyDistance"].get<float>(),
			j["neuronMaxNearbyDistance"].get<float>(),
			j["minStartingNeurons"].get<unsigned int>(),
			j["maxStartingNeurons"].get<unsigned int>(),
			j["maxNeurons"].get<unsigned int>(),
			j["useOutputNeurons"].get<bool>(),
			j["neuronStartingHealth"].get<float>(),
			j["neuronCGPOutputLowHealthChange"].get<float>(),
			j["neuronCGPOutputHighHealthChange"].get<float>(),
			j["neuronCGPOutputHealthChangeAmount"].get<float>(),
			j["neuronDeathHealth"].get<float>(),			
			j["neuronDuplicateHealth"].get<float>(),
			j["neuronDeathDuplicateHealthThresholdMultiplier"].get<float>(),
			j["neuronDuplicationHealthChange"].get<float>(),
			j["neuronDuplicationHealthReset"].get<bool>(),
			CGP::StringToActivationFunction(j["jNeuronActivationFunction"].get<std::string>()),
			j["neuronFireSpaceDeterioration"].get<float>(),
			j["neuronFireTimeDeterioration"].get<float>(),
			j["neuronFireLifetime"].get<unsigned int>(),
			j["inputNeuronFiresAge"].get<bool>(),
			j["usePreTrainSleep"].get<bool>(),
			j["usePostTrainSleep"].get<bool>(),
			j["brainXSize"].get<float>(),
			j["brainYSize"].get<float>(),
			j["brainZSize"].get<float>(),
			j["brainUseSameDimensions"].get<bool>(),
			j["brainResetBeforeProcessingInput"].get<bool>(),
			j["brainProcessingStepsBetweenInputAndOutput"].get<unsigned int>(),
			j["brainOutputsToAverageTogether"].get<unsigned int>(),
			j["brainInputsOnOneSide"].get<bool>(),
			j["brainOutputsOnOneSide"].get<bool>(),
			j["brainOutputsIgnoreEnvironmentInputs"].get<bool>(),
			j["minP"].get<float>(),
			j["maxP"].get<float>(),
			j["minConstraint"].get<float>(),
			j["maxConstraint"].get<float>(),
			j["maxNeuronAge"].get<unsigned int>(),
			dendriteInputs, dendriteOutputs,
			j["dendriteProgramNodes"].get<unsigned int>(),
			outputDendriteInputs, outputDendriteOutputs,
			j["outputDendriteProgramNodes"].get<unsigned int>(),
			axonInputs, axonOutputs,
			j["axonProgramNodes"].get<unsigned int>(),
			neuronInputs, neuronOutputs,
			j["neuronProgramNodes"].get<unsigned int>(),
			CGP::StringToUpdateEvent(j["updateEvent"].get<std::string>()),
			j["updateFrequency"].get<unsigned int>(),
			functionStringList, functionList,
			false  // Need to initialize updaters/connections
		);

		// Two values not part of the constructor:
		retVal->m_neuronDeathHealth_Original = j["neuronDeathHealth_Original"].get<float>();
		retVal->m_neuronDuplicateHealth_Original = j["neuronDuplicateHealth_Original"].get<float>();

		retVal->m_currNeuronNumber = j["currNeuronNumber"].get<unsigned int>();

		// Put the updaters in place:
		retVal->m_CGPDendriteUpdater = dendCGP;
		retVal->m_CGPOutputDendriteUpdater = outDendCGP;
		retVal->m_CGPAxonUpdater = axonCGP;
		retVal->m_CGPNeuronUpdater = neuronCGP;
		retVal->m_CGPChemicalUpdater = chemCGP;

		// Read in each input axon:
		retVal->m_inputAxons.clear();
		for (auto& elem : j["inputAxons"])
		{
			retVal->m_inputAxons.push_back(
				JAxon(elem["X"].get<float>(),
					elem["Y"].get<float>(),
					elem["Z"].get<float>()));			
		}

		// Read in each output dendrite:
		retVal->m_outputDendrites.clear();
		for (auto& elem : j["outputDendrites"])
		{
			retVal->m_outputDendrites.push_back(
				JDendrite(elem["X"].get<float>(),
					elem["Y"].get<float>(),
					elem["Z"].get<float>(),
					elem["weight"].get<float>()));
		}

		// Read in the output neurons:
		retVal->m_outputNeurons.clear();
		for (auto& elem : j["outputNeurons"])
		{
			JNeuron tmpNeuron(elem["X"].get<float>(),
				elem["Y"].get<float>(), elem["Z"].get<float>(),
				1.0, // fireValue
				1.0, // fireThreshold
				1.0, // health,
				elem["neuronNumber"].get<unsigned int>());
			tmpNeuron.m_age = elem["age"].get<unsigned int>();

			for (auto& den : elem["dendrites"])
			{
				tmpNeuron.m_dendrites.push_back(JDendrite(
					den["X"].get<float>(),
					den["Y"].get<float>(),
					den["Z"].get<float>(),
					den["weight"].get<float>()));
			}
			retVal->m_outputNeurons.push_back(tmpNeuron);
		}

		// Read in each standard neuron:
		retVal->m_neurons.clear();
		for (auto& elem : j["neurons"])
		{
			JNeuron tmpNeuron(elem["X"].get<float>(),
				elem["Y"].get<float>(), elem["Z"].get<float>(),
				elem["fireValue"].get<float>(),
				elem["fireThreshold"].get<float>(),
				elem["health"].get<float>(),
				elem["neuronNumber"].get<unsigned int>());
			
			tmpNeuron.m_fireOpportunitiesSinceLastUpdate =
				elem["fireOpportunitiesSinceLastUpdate"].get<unsigned int>();
			tmpNeuron.m_timesFiredSinceLastUpdate =
				elem["timesFiredSinceLastUpdate"].get<unsigned int>();
			tmpNeuron.m_timeStepsSinceLastFire = elem["timeStepsSinceLastFire"].get<int>();
			tmpNeuron.m_age = elem["age"].get<unsigned int>();

			// Get the neuron's axons:
			for (auto& ax : elem["axons"])
			{
				tmpNeuron.m_axons.push_back(JAxon(
					ax["X"].get<float>(),
					ax["Y"].get<float>(),
					ax["Z"].get<float>()));
			}

			// Get the dendrites:
			for (auto& den : elem["dendrites"])
			{
				tmpNeuron.m_dendrites.push_back(JDendrite(
					den["X"].get<float>(),
					den["Y"].get<float>(),
					den["Z"].get<float>(),
					den["weight"].get<float>()));
			}

			retVal->m_neurons.push_back(tmpNeuron);
		}

		return retVal;
	}

	void JBrain::setNeuronsFromStaticJson(json& neuronJson, const bool& outputNeurons)
	{
		// If this function is getting called, we assume there is at least 1 neuron.
		std::vector<JNeuron> neurons;
		for (auto& elem : neuronJson)
		{
			JNeuron tmpNeuron(elem["X"].get<float>(),
				elem["Y"].get<float>(),
				elem["Z"].get<float>(),
				elem["fireValue"].get<float>(),
				elem["fireThreshold"].get<float>(),
				elem["health"].get<float>(),
				getNextNeuronNumber());  // Ignore the provided neuron number

			tmpNeuron.m_fireOpportunitiesSinceLastUpdate = 0;
			tmpNeuron.m_timesFiredSinceLastUpdate = 0;;
			tmpNeuron.m_timeStepsSinceLastFire = 0;
			tmpNeuron.m_age = 0;

			// Get the neuron's axons:
			if (elem.contains("axons"))
			{
				for (auto& ax : elem["axons"])
				{
					tmpNeuron.m_axons.push_back(JAxon(
						ax["X"].get<float>(),
						ax["Y"].get<float>(),
						ax["Z"].get<float>()));
				}
			}

			// Get the dendrites:
			for (auto& den : elem["dendrites"])
			{
				tmpNeuron.m_dendrites.push_back(JDendrite(
					den["X"].get<float>(),
					den["Y"].get<float>(),
					den["Z"].get<float>(),
					den["weight"].get<float>()));
			}

			neurons.push_back(tmpNeuron);
		}

		// Set our gathered neurons to either standard or output:
		if (outputNeurons)
			m_outputNeurons = neurons;
		else
			m_neurons = neurons;
	}


	bool JBrain::initializeCSVOutputFile(std::string dataDirectory)
	{
		// Make sure we aren't creating dangling file pointers:
		closeCSVOutputFile();

		// We make an assumption here that each brain will only create a
		// single file in the given directory, so the data directory plus
		// our brain number is guaranteed to be unique. We also assume that the
		// function calling this properly ends the data directory with a slash:
		std::stringstream sstream;
		sstream << dataDirectory << m_name << ".csv";
		std::string filename = sstream.str();

		m_outputCSV = new std::ofstream(filename.c_str());
		if (!m_outputCSV->good())
		{
			std::cout << "Failed to open " << filename << std::endl;
			delete m_outputCSV;
			m_outputCSV = nullptr;
			return false;
		}

		// Write the first line (column headers) to the csv:
		*m_outputCSV << "neuronCount,avgNeuronHealth,dendriteCount,axonCount,score,\
sagePercent,dendMinWeight,dendMaxWeight,dendAvgWeight,neurMinFire,neurMaxFire,neurAvgFire";

		// Include the scenario's initial conditions:
		for (unsigned int i = 0; i < m_observationSize; ++i)
		{
			*m_outputCSV << ",initObs_" << i;
		}

		*m_outputCSV << std::endl;

		return true;
	}

	void JBrain::writeLineToCSVOutputFile(const float& score)
	{
		if (m_outputCSV == nullptr)
		{
			std::cout << "Trying to write to CSV output before file is opened. Brain: " 
				<< m_name << std::endl;
			return;
		}
		
		*m_outputCSV << static_cast<unsigned int>(m_neurons.size()) // Neuron count
			<< "," << getAverageNeuronHealth() // avgNeuronHealth
			<< "," << getDendriteCount()  // dendriteCount
			<< "," << getAxonCount() // axonCount
			<< "," << score; // score

		double sageMatchPercent = static_cast<double>(m_totalTrialSageChoiceMatches) /
			static_cast<double>(m_totalTrialInputsProcessed);

		*m_outputCSV << "," << sageMatchPercent;

		// Get the statistics on dendrite weights and neuron fire percentages:
		float minWeight = 0.0;
		float maxWeight = 0.0;
		float avgWeight = 0.0;
		float minFire = 0.0;
		float maxFire = 0.0;
		float avgFire = 0.0;
		getDendriteWeightStats(minWeight, maxWeight, avgWeight);
		getNeuronFirePercentages(minFire, maxFire, avgFire);

		*m_outputCSV << "," << minWeight << "," << maxWeight << "," << avgWeight;
		*m_outputCSV << "," << minFire << "," << maxFire << "," << avgFire;

		// Output the initial observation:
		for (auto& obs : m_initialObservation)
			*m_outputCSV << "," << obs;

		*m_outputCSV << std::endl;
	}

	void JBrain::closeCSVOutputFile()
	{
		// Close and delete it if it exists:
		if (m_outputCSV != nullptr)
		{			
			m_outputCSV->close();
			delete m_outputCSV;
			m_outputCSV = nullptr;
		}
	}

	void JBrain::getDendriteWeightStats(float& minWeight, float& maxWeight, float& avgWeight)
	{
		float totalWeight = 0.0;
		unsigned int dendriteCount = 0;

		// We don't want an error here if the brain killed all of its neurons:
		if (m_neurons.size() == 0)
		{
			minWeight = maxWeight = avgWeight = 0.0;
			return;
		}

		// Making reasonably safe assumption of a minimum dendrite count > 0:
		minWeight = maxWeight = m_neurons[0].m_dendrites[0].m_weight;

		for (JNeuron& neuron : m_neurons)
		{
			for (JDendrite& dend : neuron.m_dendrites)
			{
				// Gather our stats:
				totalWeight += dend.m_weight;
				++dendriteCount;				
				minWeight = fmin(minWeight, dend.m_weight);
				maxWeight = fmax(maxWeight, dend.m_weight);
			}
		}

		// Finish calculations:
		avgWeight = totalWeight / static_cast<float>(dendriteCount);
	}

	void JBrain::getNeuronFirePercentages(float& minFire, float& maxFire, float& avgFire)
	{
		minFire = maxFire = avgFire = 0.0;
		float tempFire = 0.0;
		float totalFire = 0.0;

		// No neurons? Return:
		if (m_neurons.size() == 0)
			return;
		else  // Fill in the first neuron's values as the starts:
		{
			auto neuron = m_neurons[0];
		
			// Should never get false from this check:
			if (neuron.m_fireOpportunitiesInThisRun > 0)
				tempFire = static_cast<float>(neuron.m_timesFiredInThisRun) /
				static_cast<float>(neuron.m_fireOpportunitiesInThisRun);			
			
			minFire = maxFire = tempFire;
		}		

		// For every neuron, calculate the average number of times it fired during
		// this run when it was possible to do so, and keep running stats:
		for (JNeuron& neuron : m_neurons)
		{
			tempFire = 0.0;
			if (neuron.m_fireOpportunitiesInThisRun > 0)
				tempFire = static_cast<float>(neuron.m_timesFiredInThisRun) /
				static_cast<float>(neuron.m_fireOpportunitiesInThisRun);

			minFire = fmin(tempFire, minFire);
			maxFire = fmax(tempFire, maxFire);
			totalFire += tempFire;
		}

		avgFire = totalFire / static_cast<float>(m_neurons.size());
	}

	unsigned int JBrain::getDendriteCount()
	{
		unsigned int retVal = 0;
		for (JNeuron& neuron : m_neurons)
			retVal += static_cast<unsigned int>(neuron.m_dendrites.size());

		return retVal;
	}

	unsigned int JBrain::getAxonCount()
	{
		unsigned int retVal = 0;
		for (JNeuron& neuron : m_neurons)
			retVal += static_cast<unsigned int>(neuron.m_axons.size());

		return retVal;
	}

	float JBrain::getAverageNeuronHealth()
	{
		float totalHealth = 0.0;
		if (m_neurons.size() == 0)
			return totalHealth;

		for (JNeuron& neuron : m_neurons)
			totalHealth += neuron.m_health;

		return totalHealth / static_cast<float>(m_neurons.size());
	}

	bool JBrain::setValueByName(const std::string& name, const int& value)
	{
		bool retVal = true;  // Set to false if we don't find the name

		// Long list of possible names and their values:
		if (name == "DendriteMinCount")
			m_dendriteMinCount = value;
		else if (name == "DendriteMaxCount")
			m_dendriteMaxCount = value;
		else if (name == "AxonMinCount")
			m_axonMinCount = value;
		else if (name == "AxonMaxCount")
			m_axonMaxCount = value;
		else if (name == "NeuronRefractoryPeriod")
			m_neuronRefractoryPeriod = value;		
		else if (name == "NeuronFireLifetime")
			m_neuronFireLifetime = value;		
		else if (name == "BrainProcessingStepsBetweenInputAndOutput")
			m_brainProcessingStepsBetweenInputAndOutput = value;
		else if (name == "BrainOutputsToAverageTogether")
			m_brainOutputsToAverageTogether = value;
		else if (name == "UpdateProgramFrequency")
			m_updateFrequency = value;		

		/* Not implementing mutations of available function in CGP for now.*/
		// else if (name == "CGPUseFunc_AND")
			// doStuff();
		else
			retVal = false;
				
		return retVal;
	}

	bool JBrain::setValueByName(const std::string& name, const float& value)
	{
		bool retVal = true;  // Set to false if we don't find the name

		// Long list of possible names and their values:
		if (name == "DendriteMaxLength")
			m_dendriteMaxLength = value;
		else if (name == "DendriteMinWeight")
			m_dendriteMinWeight = value;
		else if (name == "DendriteMaxWeight")
			m_dendriteMaxWeight = value;
		else if (name == "DendriteLowMoveAway")
			m_dendriteLowMoveAway = value;
		else if (name == "DendriteHighMoveToward")
			m_dendriteHighMoveToward = value;
		else if (name == "DendriteAwayTowardMoveAmount")
			m_dendriteAwayTowardMoveAmount = value;
		else if (name == "DendriteLowWeightDecrease")
			m_dendriteLowWeightDecrease = value;
		else if (name == "DendriteHighWeightIncrease")
			m_dendriteHighWeightIncrease = value;
		else if (name == "DendriteWeightChangeAmount")
			m_dendriteWeightChangeAmount = value;
		else if (name == "AxonMaxLength")
			m_axonMaxLength = value;
		else if (name == "AxonLowMoveAway")
			m_axonLowMoveAway = value;
		else if (name == "AxonHighMoveToward")
			m_axonHighMoveToward = value;
		else if (name == "AxonAwayTowardMoveAmount")
			m_axonAwayTowardMoveAmount = value;
		else if (name == "NeuronFireThreshold")
			m_neuronFireThreshold = value;
		else if (name == "NeuronDuplicateMinNearbyDistance")
			m_neuronMinNearbyDistance = value;
		else if (name == "NeuronDuplicateMaxNearbyDistance")
			m_neuronMaxNearbyDistance = value;
		else if (name == "NeuronDuplicateHealth")
			m_neuronDuplicateHealth = value;
		else if (name == "NeuronDeathHealth")
			m_neuronDeathHealth = value;
		else if (name == "NeuronStartingHealth")
			m_neuronStartingHealth = value;
		else if (name == "NeuronCGPOutputLowHealthChange")
			m_neuronCGPOutputLowHealthChange = value;
		else if (name == "NeuronCGPOutputHighHealthChange")
			m_neuronCGPOutputHighHealthChange = value;
		else if (name == "NeuronCGPOutputHealthChangeAmount")
			m_neuronCGPOutputHealthChangeAmount = value;
		else if (name == "NeuronDuplicationHealthChange")
			m_neuronDuplicationHealthChange = value;
		else if (name == "NeuronHealthThresholdMultiplier")
			m_neuronDeathDuplicateHealthThresholdMultiplier = value;
		else if (name == "NeuronMinFireValue")
			m_neuronMinFireValue = value;
		else if (name == "NeuronMaxFireValue")
			m_neuronMaxFireValue = value;
		else if (name == "NeuronFireThresholdIdleChange")
			m_neuronFireThresholdIdleChange = value;
		else if (name == "NeuronFireThresholdActiveChange")
			m_neuronFireThresholdActiveChange = value;
		else if (name == "NeuronFireSpaceDeterioration")
			m_neuronFireSpaceDeterioration = value;
		else if (name == "NeuronFireTimeDeterioration")
			m_neuronFireTimeDeterioration = value;
		else if (name == "BrainXSize")
			m_brainXSize = value;
		else if (name == "BrainYSize")
			m_brainYSize = value;
		else if (name == "BrainZSize")
			m_brainZSize = value;
		else if (name == "CGPMinConstraint")
		{
			m_minConstraint = value;
			if (m_CGPAxonUpdater != nullptr)
				m_CGPAxonUpdater->setMinConstraint(value);
			if (m_CGPDendriteUpdater != nullptr)
				m_CGPDendriteUpdater->setMinConstraint(value);
			if (m_CGPChemicalUpdater != nullptr)
				m_CGPChemicalUpdater->setMinConstraint(value);
			if (m_CGPNeuronUpdater != nullptr)
				m_CGPNeuronUpdater->setMinConstraint(value);
		}
		else if (name == "CGPMaxConstraint")
		{
			m_maxConstraint = value;
			if (m_CGPAxonUpdater != nullptr)
				m_CGPAxonUpdater->setMaxConstraint(value);
			if (m_CGPDendriteUpdater != nullptr)
				m_CGPDendriteUpdater->setMaxConstraint(value);
			if (m_CGPChemicalUpdater != nullptr)
				m_CGPChemicalUpdater->setMaxConstraint(value);
			if (m_CGPNeuronUpdater != nullptr)
				m_CGPNeuronUpdater->setMaxConstraint(value);
		}
		else if (name == "CGPMinP")
		{
			m_minP = value;
			if (m_CGPAxonUpdater != nullptr)
				m_CGPAxonUpdater->setMinP(value);
			if (m_CGPDendriteUpdater != nullptr)
				m_CGPDendriteUpdater->setMinP(value);
			if (m_CGPChemicalUpdater != nullptr)
				m_CGPChemicalUpdater->setMinP(value);
			if (m_CGPNeuronUpdater != nullptr)
				m_CGPNeuronUpdater->setMinP(value);
		}
		else if (name == "CGPMaxP")
		{
			m_maxP = value;
			if (m_CGPAxonUpdater != nullptr)
				m_CGPAxonUpdater->setMaxP(value);
			if (m_CGPDendriteUpdater != nullptr)
				m_CGPDendriteUpdater->setMaxP(value);
			if (m_CGPChemicalUpdater != nullptr)
				m_CGPChemicalUpdater->setMaxP(value);
			if (m_CGPNeuronUpdater != nullptr)
				m_CGPNeuronUpdater->setMaxP(value);
		}
		/* Not implementing mutations of available function in CGP for now.*/
		// else if (name == "CGPUseFunc_AND")
			// doStuff();
		else
			retVal = false;

		return retVal;
	}

	bool JBrain::setValueByName(const std::string& name, const std::string& value)
	{
		bool retVal = true;

		if (name == "name")
			m_name = value;
		else if (name == "parentName")
			m_parentName = value;
		else
			retVal = false;

		return retVal;
	}

	bool JBrain::setValueByName(const std::string& name, const bool& value, bool flipBool)
	{
		bool retVal = true;  // Set to false if we don't find the name

		// Long list of possible names and their values:
		if (name == "NeuronProbabilisticFire")
			if (flipBool)
				m_neuronProbabilisticFire = !m_neuronProbabilisticFire;
			else
				m_neuronProbabilisticFire = value;
		else if (name == "NeuronUseDynamicFireThresholds")
			if (flipBool)
				m_neuronUseDynamicFireThresholds = !m_neuronUseDynamicFireThresholds;
			else
				m_neuronUseDynamicFireThresholds = value;
		else if (name == "NeuronDuplicatesNearby")
			if (flipBool)
				m_neuronDuplicateNearby = !m_neuronDuplicateNearby;
			else
				m_neuronDuplicateNearby = value;
		else if (name == "NeuronDuplicationHealthReset")
			if (flipBool)
				m_neuronDuplicationHealthReset = !m_neuronDuplicationHealthReset;
			else
				m_neuronDuplicationHealthReset = value;
		else if (name == "UseOutputNeurons")
			if (flipBool)
				m_useOutputNeurons = !m_useOutputNeurons;
			else
				m_useOutputNeurons = value;
		else if (name == "InputNeuronFiresAge")
			if (flipBool)
				m_inputNeuronFiresAge = !m_inputNeuronFiresAge;
			else
				m_inputNeuronFiresAge = value;
		else if (name == "UsePreTrainSleep")
			if (flipBool)
				m_usePreTrainSleep = !m_usePreTrainSleep;
			else
				m_usePreTrainSleep = value;
		else if (name == "UsePostTrainSleep")
			if (flipBool)
				m_usePostTrainSleep = !m_usePostTrainSleep;
			else
				m_usePostTrainSleep = value;
		else if (name == "BrainInputsOnOneSide")
			if (flipBool)
				m_brainInputsOnOneSide = !m_brainInputsOnOneSide;
			else
				m_brainInputsOnOneSide = value;
		else if (name == "BrainOutputsOnOneSide")
			if (flipBool)
				m_brainOutputsOnOneSide = !m_brainOutputsOnOneSide;
			else
				m_brainOutputsOnOneSide = value;
		else if (name == "brainOutputsIgnoreEnvironmentInputs")
			if (flipBool)
				m_brainOutputsIgnoreEnvironmentInputs = !m_brainOutputsIgnoreEnvironmentInputs;
			else
				m_brainOutputsIgnoreEnvironmentInputs = value;
		else if (name == "BrainUseSameDimensions")
			if (flipBool)
				m_brainUseSameDimensions = !m_brainUseSameDimensions;
			else
				m_brainUseSameDimensions = value;
		else if (name == "ResetBeforeProcessingInput")
			if (flipBool)
				m_brainResetBeforeProcessingInput = !m_brainResetBeforeProcessingInput;
			else
				m_brainResetBeforeProcessingInput = value;
		/* Not implementing mutations of available function in CGP for now.*/
		// else if (name == "CGPUseFunc_AND")
			// doStuff();
		else
			retVal = false;

		return retVal;
	}


	void JBrain::writeSelfHumanReadable(std::ostream& out)
	{
		out << "----- Begin Brain: " << m_name << " -----" << std::endl;
		out << "\t Parent's name: " << m_parentName << std::endl;
		out << "\tDendrite Max Length: " << m_dendriteMaxLength << std::endl;
		out << "\tDendrite Count: " << m_dendriteMinCount << " - " << m_dendriteMaxCount << std::endl;
		out << "\tAxon Max Length: " << m_axonMaxLength << std::endl;
		out << "\tAxon Count: " << m_axonMinCount << " - " << m_axonMaxCount << std::endl;
		out << "\tNeuron Probabilistic Fire: " << m_neuronProbabilisticFire << std::endl;
		out << "\tNeuron Fire Threshold: " << m_neuronFireThreshold << std::endl;
		out << "\tDynamic neuron fire threshold: " << m_neuronUseDynamicFireThresholds << std::endl;
		out << "\tDynamic neuron fire changes: " << m_neuronFireThresholdIdleChange << ", " << m_neuronFireThresholdActiveChange << std::endl;
		out << "\tNeuron Fire Value Range: " << m_neuronMinFireValue << " - " << m_neuronMaxFireValue << std::endl;
		out << "\tInput neuron fires age: " << m_inputNeuronFiresAge << std::endl;
		out << "\tNeuron Refractory Period: " << m_neuronRefractoryPeriod << std::endl;
		out << "\tNeuron Duplicate Nearby: " << m_neuronDuplicateNearby << std::endl;
		out << "\tNeuron duplication health reset: " << m_neuronDuplicationHealthReset << std::endl;
		out << "\tJNeuron activation function: " << CGP::ActivationFunctionToString(m_jNeuronActivationFunction) << std::endl;
		out << "\tNeuron Nearby Distance: " << m_neuronMinNearbyDistance << " - " << m_neuronMaxNearbyDistance << std::endl;
		out << "\tUse Pre-Train Sleep: " << m_usePreTrainSleep << std::endl;
		out << "\tUse Post-Train Sleep: " << m_usePostTrainSleep << std::endl;
		out << "\tBrain Dimensions: " << m_brainXSize << " x " << m_brainYSize << " x " << m_brainZSize << std::endl;
		out << "\tBrain Use Same Dimensions: " << m_brainUseSameDimensions << std::endl;
		out << "\tUpdate programs will run after every " << m_updateFrequency << " occurences of " << CGP::UpdateEventToString(m_updateEvent) << std::endl;
		out << "\tDendrite updater CGP: " << m_CGPDendriteUpdater << std::endl;
		out << "\tAxon Updater CGP: " << m_CGPAxonUpdater << std::endl;
		out << "\tNeuron Updater CGP: " << m_CGPNeuronUpdater << std::endl;
		out << "\tChemical Updater CGP: " << m_CGPChemicalUpdater << std::endl;
		
		// Assuming out == std::cout for all CGP Program updates:
		if (m_CGPDendriteUpdater != nullptr)
		{
			out << "\t*****Dendrite Updater*****" << std::endl;
			
			// Safe to assume at least 1 input and 1 output as part of making
			// the output more pretty:
			out << "Dendrite inputs: " << CGP::CGPInputToString(m_dendriteInputs[0]);
			for (auto iter = m_dendriteInputs.begin() + 1; iter != m_dendriteInputs.end(); ++iter)
				out << ", " << CGP::CGPInputToString(*iter);
			out << std::endl;
			out << "Dendrite outputs: " << CGP::CGPOutputToString(m_dendriteOutputs[0]);
			for (auto iter = m_dendriteOutputs.begin() + 1; iter != m_dendriteOutputs.end(); ++iter)
				out << ", " << CGP::CGPOutputToString(*iter);
			out << std::endl;

			m_CGPDendriteUpdater->printGenotype();
		}

		if (m_CGPAxonUpdater != nullptr)
		{
			out << "\t*****Axon Updater*****" << std::endl;
			
			out << "Axon inputs: " << CGP::CGPInputToString(m_axonInputs[0]);
			for (auto iter = m_axonInputs.begin() + 1; iter != m_axonInputs.end(); ++iter)
				out << ", " << CGP::CGPInputToString(*iter);
			out << std::endl;
			out << "Axon outputs: " << CGP::CGPOutputToString(m_axonOutputs[0]);
			for (auto iter = m_axonOutputs.begin() + 1; iter != m_axonOutputs.end(); ++iter)
				out << ", " << CGP::CGPOutputToString(*iter);
			out << std::endl;
			m_CGPAxonUpdater->printGenotype();
		}

		if (m_CGPNeuronUpdater != nullptr)
		{
			out << "\t*****Neuron Updater*****" << std::endl;
			
			out << "Neuron inputs: " << CGP::CGPInputToString(m_neuronInputs[0]);
			for (auto iter = m_neuronInputs.begin() + 1; iter != m_neuronInputs.end(); ++iter)
				out << ", " << CGP::CGPInputToString(*iter);
			out << std::endl;
			out << "Neuron outputs: " << CGP::CGPOutputToString(m_neuronOutputs[0]);
			for (auto iter = m_neuronOutputs.begin() + 1; iter != m_neuronOutputs.end(); ++iter)
				out << ", " << CGP::CGPOutputToString(*iter);
			out << std::endl;

			m_CGPNeuronUpdater->printGenotype();
		}

		if (m_CGPChemicalUpdater != nullptr)
		{
			out << "\t*****Chemical Updater*****" << std::endl;
			m_CGPChemicalUpdater->printGenotype();
		}
		
		out << "*** Begin neurons of brain " << m_name << " ***" << std::endl;
		for (unsigned int i = 0; i < m_neurons.size(); ++i)
		{
			m_neurons[i].writeSelfHumanReadable(out);
		}

		out << "----- End Brain " << m_name << " -----" << std::endl;
	}
	
	bool operator==(const NeuronFire& lhs, const NeuronFire& rhs)
	{
		if (&lhs == &rhs)
			return true;

		return isEqual(lhs, rhs) &&
			lhs.m_age == rhs.m_age &&
			lhs.m_environmentInput == rhs.m_environmentInput &&
			fabs(lhs.m_fireValue - rhs.m_fireValue) < FLT_EPSILON;
	}

	bool isEqual(const JBrainComponent& lhs, const JBrainComponent& rhs)
	{
		return fabs(lhs.m_X - rhs.m_X) < FLT_EPSILON &&
			fabs(lhs.m_Y - rhs.m_Y) < FLT_EPSILON &&
			fabs(lhs.m_Z - rhs.m_Z) < FLT_EPSILON;
	}

	// Comparison operators must be non-class members to make vector
	// comparisons legal:
	bool operator==(const JDendrite& lhs, const JDendrite& rhs)
	{
		if (&lhs == &rhs)
			return true;

		// Compare the values saved to file, not those calculated at each step: the vector of inputs provided:
		
		return isEqual(lhs, rhs) &&
			(fabs(lhs.m_weight - rhs.m_weight) < FLT_EPSILON);
	}

	bool operator==(const JAxon& lhs, const JAxon& rhs)
	{
		if (&lhs == &rhs)
			return true;

		// Compare all class variables that get saved to file:
		return isEqual(lhs, rhs);				
	}

	bool operator==(const JNeuron& lhs, const JNeuron& rhs)
	{
		if (&lhs == &rhs)
			return true;

		// Compare all class variables that are saved to file.
		// Don't compare those calculated at each step:
		return isEqual(lhs, rhs) &&
			(lhs.m_axons == rhs.m_axons) &&
			(lhs.m_dendrites == rhs.m_dendrites) &&
			(fabs(lhs.m_fireValue - rhs.m_fireValue) < FLT_EPSILON) &&
			(fabs(lhs.m_fireThreshold - rhs.m_fireThreshold) < FLT_EPSILON) &&
			(fabs(lhs.m_health - rhs.m_health) < FLT_EPSILON) &&
			(lhs.m_neuronNumber == rhs.m_neuronNumber) &&
			(lhs.m_age == rhs.m_age);
	}

	float euclideanDist(const float& x1, const float& y1, const float& z1,
		const float& x2, const float& y2, const float& z2)
	{
		float dx = fabsf(x1 - x2);
		float dy = fabsf(y1 - y2);
		float dz = fabsf(z1 - z2);

		// sqrt (dx2 + dy2 + dz2)
		return powf((dx * dx) + (dy * dy) + (dz * dz), 0.5);
	}
}  // End namespace JBrain