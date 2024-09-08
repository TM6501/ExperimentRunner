#pragma once

#include "JBrainCGPIndividual.h"
#include "JBrainComponents.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include "Enums.h"
#include "json.hpp"
using json = nlohmann::json;

namespace JBrain
{
	// Struct to hold where/when a neuron's axon fired:
	struct NeuronFire : public JBrainComponent
	{
		int m_age;
		float m_fireValue;
		
		// If true, this fire represents an environmental input. It is special-cased
		// in several locations:
		bool m_environmentInput;

		NeuronFire& operator=(const NeuronFire& rhs)
		{
			if (this == &rhs)
				return *this;

			setEqual(rhs);
			m_age = rhs.m_age;
			m_fireValue = rhs.m_fireValue;
			m_environmentInput = rhs.m_environmentInput;
			return *this;
		}

		friend bool operator==(const NeuronFire& lhs, const NeuronFire& rhs);		

		// Age defaults to -1 because these fires will be created, then their
		// age incremented before being checkd as inputs to any dendrites.
		NeuronFire(const float& x, const float& y, const float& z,
			const float& fireValue, const bool& environmentInput = false) :
			JBrainComponent(x, y, z), m_age(-1), m_fireValue(fireValue),
			m_environmentInput(environmentInput)
		{}
	};
	
	class JBrain
	{
		// To allow mutation-specific changes:
		friend class JBrainFactory;

	private:
		// Give each brain a name for debugging purposes:
		std::string m_name;
		std::string m_parentName;

		// Environment action and observation sizes. Eventually, this
		// will become mutable:
		unsigned int m_observationSize;
		unsigned int m_actionSize;

		// Inputs to the brain (from the environment) are disembodied axons. The
		// outputs are disembodied dendrites.
		std::vector<JDendrite> m_outputDendrites;
		std::vector<JAxon> m_inputAxons;

		// Dendrites:
		float m_dendriteMinLength;
		float m_dendriteMaxLength;
		unsigned int m_dendriteMinCount;
		unsigned int m_dendriteMaxCount;
		float m_dendriteMinWeight;
		float m_dendriteMaxWeight;

		float m_dendriteLowMoveAway;
		float m_dendriteHighMoveToward;
		float m_dendriteAwayTowardMoveAmount;

		float m_dendriteLowWeightDecrease;
		float m_dendriteHighWeightIncrease;
		float m_dendriteWeightChangeAmount;

		// Axons:
		float m_axonMinLength;
		float m_axonMaxLength;
		unsigned int m_axonMinCount;
		unsigned int m_axonMaxCount;
		float m_axonLowMoveAway;
		float m_axonHighMoveToward;
		float m_axonAwayTowardMoveAmount;

		// Neurons:
		bool m_neuronProbabilisticFire;  // If false, threshold fire is used.
		float m_neuronFireThreshold;  // Ignored if probabilistic fire is used.
		float m_neuronMinFireValue;  // The min and max values that a neuron will
		float m_neuronMaxFireValue;  // generate when firing.
		bool m_neuronUseDynamicFireThresholds;
		float m_neuronFireThresholdIdleChange;  // How much the fire threshold of
		float m_neuronFireThresholdActiveChange;  // a neuron changes when idle or firing
		unsigned int m_neuronRefractoryPeriod;
		bool m_neuronDuplicateNearby;
		float m_neuronMinNearbyDistance;  // Ignored if not duplicating nearby.
		float m_neuronMaxNearbyDistance;  // Ignored if not duplicating nearby.
		unsigned int m_minStartingNeurons;
		unsigned int m_maxStartingNeurons;
		unsigned int m_maxNeurons;

		// The health thresholds to trigger neuron death or duplication and the
		// multiplier that changes them when either event occurs:
		float m_neuronDeathHealth;
		float m_neuronDeathHealth_Original;
		float m_neuronDuplicateHealth;
		float m_neuronDuplicateHealth_Original;
		float m_neuronDeathDuplicateHealthThresholdMultiplier;
		
		// Getting duplicated changes a neuron's health value:
		float m_neuronDuplicationHealthChange;
		// If True, ignore m_neuronDuplicationHealthChange and instead
		// reset the duplicated neuron's health back to a value within
		// the random starting health values:
		bool m_neuronDuplicationHealthReset;

		// The activation function used by each JNeuron:
		CGP::JNEURON_ACTIVATION_FUNCTION m_jNeuronActivationFunction;

		// Tracking where/when neurons fired and how their values are read in
		// from other dendrites:
		std::vector<NeuronFire> m_neuronFires;
		float m_neuronFireSpaceDeterioration;
		float m_neuronFireTimeDeterioration;
		unsigned int m_neuronFireLifetime;
		bool m_inputNeuronFiresAge;

		// To keep track of all of the neurons created, even if they've been
		// destroyed:
		unsigned int m_currNeuronNumber;
		unsigned int getNextNeuronNumber();
		
		// Sleep:
		// With standard RL environments, these make work out to be the same thing and could cause
		// a double sleep. The only real change then, would be whether or not there is a sleep cycle
		// directly before the first training session.
		bool m_usePreTrainSleep;  // Put the brain through a sleep cycle directly pre-training.
		bool m_usePostTrainSleep;  // Put the brain through a sleep cycle directly post-training.

		// Brain:
		float m_brainXSize;
		float m_brainYSize;
		float m_brainZSize;		
		bool m_brainUseSameDimensions;  // If true, all three dimensions will be forced to stay the same.		                                
		bool m_brainResetBeforeProcessingInput;

		// Inputs and outputs are either on the sides or spaced throughout
		// the brain:
		bool m_brainInputsOnOneSide;
		bool m_brainOutputsOnOneSide;

		// How many brain time steps happen between when the environment values
		// are provided to the brain and output is read from the brain.
		unsigned int m_brainProcessingStepsBetweenInputAndOutput;

		// Circuits (Nothing circuit-related is implemented, yet):
		float m_circuitMinDimensions;
		float m_circuitMaxDimensions;
		bool m_circuitUseSameDimensions;  // If true, all three dimensions will be the same.
		unsigned int m_circuitMinCircuitCount;
		unsigned int m_circuitMaxCircuitCount;  // Setting min and max count to 0 will effectively turn off circuits.
		float m_circuitProbabilityCircuitPassedToChild;
		float m_circuitProbabilityFireChangeWhenOtherNeuronFires;
		float m_circuitProbabilityNeuronDuplicateInCircuit;  // This value is checked first. If false, then duplicate-nearby is considered.
		float m_circuitNeuronHealthChangeFromNeuronDeath;
		float m_circuitNeuronHealthChangeFromNeuronDuplication;
		bool m_circuitsCanOverlap;

		// Equations that dictate changes in the individual neuron variables.
		// Each brain will maintain a monolithic update function as well as several
		// smaller functions each responsible for a different aspect of the update.
		// Mutation may change the brain from monolithic to separated functions, effectively
		// re-activating dormant DNA. This is important for biological realism.
		bool m_equationUseMonolithic;

		// P is a static variable value input into each function. It is chosen
		// randomly (and mutated) at the individual CGP-node level. These
		// values represent the minimum and maximum values P can take on
		// in the CGP programs:
		float m_minP;
		float m_maxP;

		// Constraints are used to make sure values don't explode while processing.
		// Every value that is returned from a CGP node will be constrained between
		// these two values:
		float m_minConstraint;
		float m_maxConstraint;

		// The average number of times the neurons fired (when given the opportunity)
		// since the last update:
		float m_averageNeuronFirePercentage;

		// How often, since the last update, our output matched the sage's output:
		float m_sageMatchPercent;

		// The maximum age of a neuron to use when calculating their age. "Age"
		// needs to be a floating point value. Once the age is beyond this number,
		// that floating point value will be 1.0:
		unsigned int m_maxNeuronAge;

		// The health of any randomly created or created-through-duplication neuron:
		float m_neuronStartingHealth;

		// If health-up-down is used rather than an absolute change, these variables
		// help dictate how health changes:
		float m_neuronCGPOutputLowHealthChange;
		float m_neuronCGPOutputHighHealthChange;
		float m_neuronCGPOutputHealthChangeAmount;
		
		// Singular Updaters:
		CGP::JBrainCGPIndividual* m_CGPDendriteUpdater;
		CGP::JBrainCGPIndividual* m_CGPAxonUpdater;
		CGP::JBrainCGPIndividual* m_CGPNeuronUpdater;
		CGP::JBrainCGPIndividual* m_CGPChemicalUpdater;
		bool areEqual(CGP::JBrainCGPIndividual* lhs, CGP::JBrainCGPIndividual* rhs);

		// The inputs for each CGP program:
		std::vector<CGP::CGP_INPUT> m_dendriteInputs;
		std::vector<CGP::CGP_OUTPUT> m_dendriteOutputs;
		std::vector<CGP::CGP_INPUT> m_axonInputs;
		std::vector<CGP::CGP_OUTPUT> m_axonOutputs;
		std::vector<CGP::CGP_INPUT> m_neuronInputs;
		std::vector<CGP::CGP_OUTPUT> m_neuronOutputs;

		// The number of CGP nodes to allocate to each program:
		unsigned int m_dendriteProgramNodes;
		unsigned int m_axonProgramNodes;
		unsigned int m_neuronProgramNodes;

		// Use our class variables to create each of the singular updaters.
		void createDendriteUpdater();
		void createAxonUpdater();
		void createNeuronUpdater();
		void createChemicalUpdater();
		void createAllSeparateUpdaters(); // Call the 4 other creation functions.

		// Monolithic Updater (Not used for now):
		CGP::JBrainCGPIndividual* m_CGPMonolithicUpdater;

		// Update frequency:
		CGP::UPDATE_EVENT m_updateEvent;
		unsigned int m_updateFrequency;

		// Equations available to the JBrainCGPIndividuals:
		std::vector<std::string> m_functionStringList;
		std::vector<std::function<double(double, double, double)> > m_functionList;

		std::vector<JNeuron> m_neurons;

		// Records of the choices this brain made and choices made by
		// the sage (if used). This is used to calculate the percentage of time
		// we make the "right" choice:
		std::vector<int> m_sageChoices;
		std::vector<int> m_brainChoices;
		unsigned int m_inputProcessingsSinceLastUpdate;

		// Data recorded for a full trial:
		std::vector<double> m_initialObservation;
		unsigned int m_totalTrialInputsProcessed;
		unsigned int m_totalTrialSageChoiceMatches;

		// Our output csv file:
		std::ofstream* m_outputCSV;

		// Used by processInput, it moves the brain one step forward in time 
		// by processing all dendrite inputs and firing axons if needed.
		void singleTimeStepForward();
		
		// Apply the changes to the fire value associated with deterioration.
		// References are used to keep them as fast as possible:
		void applyDistanceDeterioration(float& fireValue, const float& distance);
		void applyTimeDeterioration(float& fireValue, const float& time);

		// Utility functions that would be in a separate utility file if 
		// I were a less lazy programmer:
		float getRandomFloat(const float& min, const float& max);
		int getRandomInt(const int& min, const int& max);
		float getDistance(const float& x1, const float& y1, const float& z1,
			              const float& x2, const float& y2, const float& z2);
		float getDistance(const JBrainComponent& a, const JBrainComponent& b);

		// Add random parts to a neuron:
		void addRandomAxon(JNeuron& neuron);
		void addRandomDendrite(JNeuron& dendrite);

		// Calculate how often this brain's output matched the sage's output.
		// Return a value between 0.0 and 1.0:
		void calculateSageMatch();

		// Calculate statistics about dendrite weights and neuron
		// fire percentages:
		void getDendriteWeightStats(float& minWeight, float& maxWeight, float& avgWeight);

		// Calculate statistics about neuron fire percentages:
		void getNeuronFirePercentages(float& minFire, float& maxFire, float& avgFire);

		// Create the input axon and output dendrites that the brain uses
		// to send and receive from the outside world.
		void createInputsAndOutputs();
		
		// Given the input coordinates, produce a 3-coordinate vector within
		// the available distance which fits inside the brain:
		std::vector<float> getValidCoordinatesWithinDistance(const float& startX,
			const float& startY, const float& startZ, const float& distance);

		// Get the input to a given dendrite, taking into account time
		// and distance deterioration as well as its own weight. This will
		// update the dendrite's most recent input values and location of its
		// biggest contributor:
		float getDendriteInput(JDendrite& dendrite);

		// Check if neuron fires:
		bool getIfNeuronFires(JNeuron& neuron);

		// The steps taken by this single timestep:
		// 1. Increment the age of all neuron fires and times since the neurons
		//    last fired:
		void incrementAllNeuronFireAges();  // Increment and remove too old.
		void allNeuronsSingleTimeStepForward();  // Increment ages, time since fired
		
		// 2. Go through each neuron, check if it should fire, and fire it if so:
		void fireAllNeurons();

		// The steps in processing input:
		void resetBrainForNewInputs();
		void setAllInputAxonFires(const std::vector<double>& inputs);
		// Take the appropriate number of single time steps forward
		std::vector<double> readBrainOutputs();

		// Take the total input from a neuron's dendrites and apply the
		// selected activation function:
		float applyJNeuronActivationFunction(const float& input);

		// Run any post-brain-processing updates. This may involve running
		// the CGP update functions:
		void updateAfterProcessingInput();
		
		// Part of the update processing. If dendrites are using the nearest
		// axon as part of their input, we need to know it. Since it will change
		// with every update, it must be recalculated before every update. The
		// same is true in the reverse:
		void calculateNearestAxonToEveryDendrite();
		void calculateNearestDendriteToEveryAxon();

		// Determine the average number of times the neurons in the brain fired:
		void calculateAverageNeuronFirePercentage();

		// ************* We need to be careful that all CGP inputs are static
		// and remain that way through all updates.  Don't calculate any distances
		// based on changing values, for instance.
		void applyAllCGP();

		// Every brain part update will consist of gathering the inputs, 
		// running the CGP updater, and apply its outputs:
		void applyCGP(JNeuron& neuron);
		std::vector<double> getCGPInputs(const JNeuron& neuron);
		void applyCGPOutputs(JNeuron& neuron, const std::vector<double>& cgpOutputs);

		// Dendrite version requires the parent neuron:
		void applyCGP(JDendrite& dendrite, const JNeuron& parentNeuron);
		std::vector<double> getCGPInputs(const JDendrite& dendrite, const JNeuron& parentNeuron);
		void applyCGPOutputs(JDendrite& dendrite, const std::vector<double>& cgpOutputs,
			const JNeuron& parentNeuron);

		void applyCGP(JAxon& axon, const JNeuron& parentNeuron);
		std::vector<double> getCGPInputs(const JAxon& axon, const JNeuron& parentNeuron);
		void applyCGPOutputs(JAxon& axon, const std::vector<double>& cgpOutputs,
			const JNeuron& parentNeuron);

		// With neuron health values all updated, kill/duplicate those that have
		// crossed those thresholds.
		void duplicateAndKillNeurons();
		JNeuron createDuplicateNeuron(JNeuron& neuron);
		JNeuron createNewNeuron(const float& x, const float& y, const float& z);

		// Gather status values:
		unsigned int getDendriteCount();
		unsigned int getAxonCount();
		float getAverageNeuronHealth();
		
	public:
		inline std::string getName() { return m_name; }
		inline std::string getParentName() { return m_parentName; }
		inline unsigned int getNeuronCount()
		{ return static_cast<unsigned int>(m_neurons.size()); }

		// Get/Set the JNeuron activation function:
		inline CGP::JNEURON_ACTIVATION_FUNCTION getJNeuronActivationFunction() { return m_jNeuronActivationFunction; }
		inline void setValue(CGP::JNEURON_ACTIVATION_FUNCTION newActFun) { m_jNeuronActivationFunction = newActFun; }

		// For recording data. Later, we may make the data columns
		// configurable in the yaml, but for the sake of speed, they
		// will be hard coded for now:
		bool initializeCSVOutputFile(std::string dataDirectory);
		void writeLineToCSVOutputFile(const float& score);
		void closeCSVOutputFile();

		// If the size of the brain changed, some things may no longer make
		// sense. This function handles the logical inconsistencies:
		void handleBrainSizeChange();
		
		// Create 1 or more neurons with random valid parameters and add
		// them to our  vector of neurons.
		void addRandomStartingNeurons(bool destroyCurrentNeurons=false);

		// Take the inputs from an environment and produce outputs. The sage's
		// choice is the argMax of the output and only functions properly when
		// the environment in question is a single choice output, like CartPole.
		// Leaving the choice as -1 indicates that it isn't provided.
		// The brain records certain data at the start of a new test trial
		// in a given environment, newTrial=true should be set at the beginning
		// of any trial run.
		std::vector<double> processInput(const std::vector<double>& inputs,
			int sageChoice=-1, bool newTrial=false);

		bool operator==(const JBrain& rhs);

		// flipBool -> Ignore the value if this is a boolean. Negate the boolean.
		bool setValueByName(const std::string& name, const float& value);
		bool setValueByName(const std::string& name, const int& value);
		bool setValueByName(const std::string& name, const bool& value, bool flipBool);
		bool setValueByName(const std::string& name, const std::string& value);

		~JBrain();
		JBrain(const JBrain& other);
		JBrain(const std::string& name,
			const std::string& parentName,
			const unsigned int& observationSize,
			const unsigned int& actionSize,
			const float& dendriteMinLength,
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
			const float& axonMinLength,
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
			const bool& brainInputsOnOneSide,
			const bool& brainOutputsOnOneSide,
            const float& circuitMinDimensions,
			const float& circuitMaxDimensions,
		    const bool& circuitUseSameDimensions,
			const unsigned int& circuitMinCircuitCount,
			const unsigned int& circuitMaxCircuitCount,
		    const float& circuitProbabilityCircuitPassedToChild,
		    const float& circuitProbabilityFireChangeWhenOtherNeuronFires,
		    const float& circuitProbabilityNeuronDuplicateInCircuit,
			const float& circuitNeuronHealthChangeFromNeuronDeath,
		    const float& circuitNeuronHealthChangeFromNeuronDuplication,
		    const bool& circuitsCanOverlap,
			const bool& equationUseMonolithic,
			const float& minP, const float& maxP,
			const float& minConstraint, const float& maxConstraint,
			const unsigned int& maxNeuronAge,
			const std::vector<CGP::CGP_INPUT>& dendriteInputs,
			const std::vector<CGP::CGP_OUTPUT>& dendriteOutputs,
			const unsigned int& dendriteProgramNodes,
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
			const bool& needInitializeUpdatersAndConnections = true);
		
		JBrain(std::string yamlFilename);
		void writeSelfToJson(json& j);
		static JBrain* getBrainFromJson(json& j);
		void writeSelfHumanReadable(std::ostream& out);
	};

} // End namespace JBrain