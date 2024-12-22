#pragma once

#include "JBrainCGPIndividual.h"
#include "JBrainComponents.h"
#include "ObservationProcessor.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include "Enums.h"
#include "JsonLib\json.hpp"
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
	
	class JBrain_Snap
	{
		// to allow mutation-specific changes:
		friend class JBrainFactory;

	private:
		// Calculate various chances:
		double getChance_CorrectGotInput_IncreaseWeight();
		double getChance_CorrectGotInput_CreateConnection();
		double getChance_YesFired_UnusedInput_DecreaseWeight();
		double getChance_YesFired_UnusedInput_BreakConnection();
		double getChance_WrongGotInput_DecreaseWeight();
		double getChance_WrongGotInput_BreakConnection();
		double getChance_Step_CreateProcessingNeuron();
		double getChance_Step_DestroyProcessingNeuron();
		double getChance_Step_CreateInputNeuron();
		double getChance_Step_DestroyInputNeuron();
		double getChance_Run_CreateProcessingNeuron();
		double getChance_Run_DestroyProcessingNeuron();
		double getChance_Run_CreateInputNeuron();
		double getChance_Run_DestroyInputNeuron();
		double getChance_NoOut_IncreaseInputDendriteWeight();
		double getChance_NoOut_AddProcessingNeuronDendrite();
		double getChance_NoOut_IncreaseProcessingNeuronDendriteWeight();
		double getChance_NoOut_AddOutputNeuronDendrite();
		double getChance_NoOut_IncreaseOutputNeuronDendriteWeight();
		double getChance_NoOut_CreateProcessingNeuron();

		unsigned int getNeuronCount(); // Get a count of non-null neurons we have

		std::string m_name;
		std::string m_parentName;
		std::vector<double> m_mostRecentBrainInputs;

		double m_staticOverallProbability; // Passed in constructor.
		double m_overallProbability; // Calculated using dynamic probability.
		CGP::DYNAMIC_PROBABILITY m_dynamicProbabilityUsage;
		double m_dynamicProbabilityMultiplier;
		double m_mostRecentScorePercent;
		unsigned int m_neuronAccumulateDuration;
		bool m_neuronResetOnFiring;
		bool m_neuronResetAfterOutput;
		double m_neuronFireThreshold;
		unsigned int m_neuronMaximumAge;
		unsigned int m_brainProcessingStepsAllowed;

		// Stored so we can pass them in a copy constructor:
		unsigned int m_initialInputNeuronCount;
		unsigned int m_initialProcessingNeuronCount;
		unsigned int m_maximumProcessingNeuronCount;
		unsigned int m_maximumInputNeuronsToInputRatio;

		// Dendrite-specific variables:
		double m_dendriteWeightChange;
		double m_dendriteMinimumWeight;
		double m_dendriteMaximumWeight;
		double m_dendriteStartingWeight;
		unsigned int m_dendriteMinCountPerNeuron;
		unsigned int m_dendriteMaxCountPerNeuron;
		unsigned int m_dendriteStartCountPerNeuron;

		// Base neuron count is used to regulate the neuron-creation and neuron-destruction chances:
		unsigned int m_baseProcessingNeuronCount;

		// Environment-related values:
		unsigned int m_observationSize;
		unsigned int m_actionSize;
		int m_correctOutputNeuron;  // The neuron we WANT to fire, based on the sage choice.

		// Step-Events:
		double m_stepCreateNeuronChance;
		double m_stepCreateNeuron_BaseCountRatioMultiplier;
		double m_stepCreateInputNeuronChance;
		double m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier;
		double m_stepDestroyNeuronChance;
		double m_stepDestroyNeuron_CountBaseRatioMultiplier;
		double m_stepDestroyInputNeuronChance;
		double m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier;

		// Weighting the processing neuron destruction:
		bool m_destroyNeuron_FavorFewerConnections;
		bool m_destroyNeuron_FavorYoungerNeurons;

		// Run-Events:
		double m_runCreateNeuronChance;
		double m_runCreateNeuron_BaseCountRatioMultiplier;
		double m_runCreateInputNeuronChance;
		double m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier;
		double m_runDestroyNeuronChance;
		double m_runDestroyNeuron_CountBaseRatioMultiplier;
		double m_runDestroyInputNeuronChance;
		double m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier;

		// Output-Positive Events:
		double m_outputPositive_CascadeProbability;
		double m_outputPositive_InSequence_IncreaseDendriteWeight;
		double m_outputPositive_NoConnection_InSequence_CreateConnection;
		double m_outputPositive_YesFire_UnusedInput_DecreaseWeight;
		double m_outputPositive_YesFire_UnusedInput_BreakConnection;

		// Output-Negative Events:
		double m_outputNegative_CascadeProbability;
		double m_outputNegative_InSequence_DecreaseDendriteWeight;
		double m_outputNegative_InSequence_BreakConnection;

		// No Output events:
		double m_noOutput_IncreaseInputDendriteWeight;
		double m_noOutput_AddProcessingNeuronDendrite;
		double m_noOutput_IncreaseProcessingNeuronDendriteWeight;
		double m_noOutput_AddOutputNeuronDendrite;
		double m_noOutput_IncreaseOutputNeuronDendriteWeight;
		double m_noOutput_CreateProcessingNeuron;

		// A single observation processor pointer is held by all brains. The factory allocates it for us:
		ObservationProcessor* m_observationProcessor;

		// Easy function to check if an event occured:
		bool getEventHappened(double probability);

		// Get a count of how much each input is used:
		std::vector<unsigned int> getUsedInputsCount();

		// Get a count of how much the output of each neuron is used:
		std::vector<unsigned int> getOutputCountVector(const std::vector<JNeuron_Snap*> checkVec);

		// Create all of our initial neurons:
		void createAllStartingNeurons(const unsigned int& inputCount, const unsigned int& processingNeurons);

		// Sanity check and initial neuron creator:
		void ensureAllInputsUsed(); // Create input neurons until all of the inputs are used at least once.
		bool getInsideMaximumInputNeuronToInputsRatio();
		bool getInsideMaximumProcessingNeuronsCount();

		// Random events:
		void handleStepEvents();
		void handleEndOfRunEvents();
		void handleNoOutputEvents();

		// Implementation of events:
		void doCreateProcessingNeuron();
		void doDestroyProcessingNeuron();
		void doCreateInputNeuron();
		void doDestroyInputNeuron();
		void doCreateOutputNeuron();
		void doDecreaseDendriteWeight(const unsigned int& neuronNumber, const unsigned int& inputNeuronNumber);
		void doDropDendriteConnection(const unsigned int& neuronNumber, const unsigned int& inputNeuronNumber);

		// Events that only occur when the brain fails to produce output:
		void doAddOutputNeuronDendrite(JNeuron_Snap* outputNeuron);
		void doAddProcessingNeuronDendrite(JNeuron_Snap* procNeuron);

		// Recursive functions that cascade through contributing neurons:
		void handleCorrectOutputNeuronGotInputEvent(const unsigned int& neuron, const unsigned int& stepNumber,
			double increaseWeightChance = -1.0, double createConnectionChance = -1.0);
		void handleWrongOutputNeuronGotInputEvent(const unsigned int& neuron, const unsigned int& stepNumber,
			double decreaseWeightChance = -1.0, double breakConnectionChance = -1.0);
		void handleCorrectOutputNeuronFiredEvent(const unsigned int& neuron, const unsigned int& stepNumber);

		// Non-recursive special-casing for input neurons:
		void doHandleInputNeuronIncreaseWeights(const unsigned int& neuron);
		void doHandleInputNeuronCreateConnection(const unsigned int& neuron);
		void doHandleInputNeuronDecreaseWeights(const unsigned int& neuron);
		void doHandleInputNeuronBreakConnection(const unsigned int& neuron);

		// Get lists of neurons:
		std::vector<JNeuron_Snap*> getAllNeuronsFiredOnStep(const unsigned int& stepNumber);

		// This vector holds all neurons, in order and will store nullptrs when neurons are deleted.
		// This is to make for easy access by ensuring that m_allNeurons[X] is always neuron number X.
		std::vector<JNeuron_Snap*> m_allNeurons;
		std::vector<JNeuron_Snap*> m_inputNeurons;
		std::vector<JNeuron_Snap*> m_processingNeurons; // Need to be kept in order of outputs
		std::vector<JNeuron_Snap*> m_outputNeurons;

		// Remove all references to this neuron, then delete it. Remove it from the neuron storage
		// vectors except for m_allNeurons which will store a nullptr in its place.
		void deleteNeuron(unsigned int neuronNumber);

		// Check if a given neuron should fire or not. If it does, set this step number in the fire steps.
		bool setIfInputNeuronFired(const unsigned int& neuronNumber, const int& currentStepNumber);
		bool setIfNonInputNeuronFired(const unsigned int& neuronNumber, const int& currentStepNumber,
																	const unsigned int& minAccumulateStep, const unsigned int& maxAccumulateStep);


		std::ofstream* m_outputCSV;
		unsigned int m_correctNeuronFiredCount;
		unsigned int m_wrongNeuronFiredCount;
		unsigned int m_brainOutputZeroFiredCount;
		unsigned int m_brainOutputOneFiredCount;
		unsigned int m_inputNeuronFiredCount;
		unsigned int m_processingNeuronFiredCount;
		unsigned int m_outputNeuronFiredCount;
		unsigned int m_inputNeuronCreatedCount;
		unsigned int m_inputNeuronDestroyedCount;
		unsigned int m_processingNeuronCreatedCount;
		unsigned int m_processingNeuronDestroyedCount;
		unsigned int m_correctOutNeuGotInputCalledCount; // Only non-recursive calls.
		unsigned int m_wrongOutNeuGotInputCalledCount;  // Only non-recursive calls.
		unsigned int m_noOutputHappenedCount;
		// End logging

	public:
		~JBrain_Snap();
		JBrain_Snap(
			const std::string& name,
			const std::string& parentName,
			const double& overallProbability,
			const CGP::DYNAMIC_PROBABILITY& dynamicProbabilityUsage,
			const double& dynamicProbabilityMultiplier,
			const unsigned int& neuronAccumulateDuration,
			const bool& neuronResetOnFiring,
			const bool& neuronResetAfterOutput,
			const double& neuronFireThreshold,
			const unsigned int& neuronMaximumAge,
			const unsigned int& brainProcessingStepsAllowed,
			const double& dendriteWeightChange,
			const double& dendriteMinimumWeight,
			const double& dendriteMaximumWeight,
			const double& dendriteStartingWeight,
			const unsigned int& dendriteMinCountPerNeuron,
			const unsigned int& dendriteMaxCountPerNeuron,
			const unsigned int& dendriteStartCountPerNeuron,
			const unsigned int& baseProcessingNeuronCount,
			const unsigned int& actionSize,
			const unsigned int& initialInputNeuronCount,
			const unsigned int& initialProcessingNeuronCount,
			const unsigned int& maximumProcessingNeuronCount,
			const unsigned int& maximumInputNeuronToInputsRatio,
			const double& stepCreateNeuronChance,
			const double& stepCreateNeuron_BaseCountRatioMultiplier,
			const double& stepCreateInputNeuronChance,
			const double& stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			const double& stepDestroyNeuronChance,
			const double& stepDestroyNeuron_CountBaseRatioMultiplier,
			const double& stepDestroyInputNeuronChance,
			const double& stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier,
			const bool& destroyNeuron_FavorFewerConnections,
			const bool& destroyNeuron_FavorYoungerNeurons,
			const double& runCreateNeuronChance,
			const double& runCreateNeuron_BaseCountRatioMultiplier,
			const double& runCreateInputNeuronChance,
			const double& runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			const double& runDestroyNeuronChance,
			const double& runDestroyNeuron_CountBaseRatioMultiplier,
			const double& runDestroyInputNeuronChance,
			const double& runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier,
			const double& outputPositive_CascadeProbability,
			const double& outputPositive_InSequence_IncreaseDendriteWeight,
			const double& outputPositive_NoConnection_InSequence_CreateConnection,
			const double& outputPositive_YesFire_UnusedInput_DecreaseWeight,
			const double& outputPositive_YesFire_UnusedInput_BreakConnection,
			const double& outputNegative_CascadeProbability,
			const double& outputNegative_InSequence_DecreaseDendriteWeight,
			const double& outputNegative_InSequence_BreakConnection,
			const double& noOutput_IncreaseInputDendriteWeight,
			const double& noOutput_AddProcessingNeuronDendrite,
			const double& noOutput_IncreaseProcessingNeuronDendriteWeight,
			const double& noOutput_AddOutputNeuronDendrite,
			const double& noOutput_IncreaseOutputNeuronDendriteWeight,
			const double& noOutput_CreateProcessingNeuron,
			ObservationProcessor* observationProcessor);

		std::vector<double> processInput(const std::vector<double>& inputs, int sageChoice);
		void getFullTrialStatistics(unsigned int& noOutputEvents, unsigned int& goodOutputs, unsigned int& badOutputs,
			unsigned int& procNeuronCreated, unsigned int& procNeuronDestroyed,
			unsigned int& inputNeuronCreated, unsigned int& inputNeuronDestroyed);
		void processEndOfTrial(double reward, const double& minReward, double maxReward); // Indicate that was the last step in the trial and let the brain respond.
		bool runProcessingSteps(std::vector<double>& brainOutputs);
		std::vector<double> readBrainOutput(const unsigned int& stepNumber);
		bool fireAllProcessingAndOutputNeurons(const unsigned int& stepNumber); // True if an output neuron fired.

		void resetAllNeuronFires();

		// Logging information:
		void initializeCSVOutputFile(std::string dataDirectory);
		void writeLineToCSVOutputFile(const double& score);
		void closeCSVOutputFile();
		void resetAllLoggingValues();
		void writeSelfToJson(json& j);

		// Used in mutating brains:
		JBrain_Snap(const JBrain_Snap& other);
		std::string getName() { return m_name; }
		std::string getParentName() { return m_parentName;}
		bool setValueByName(const std::string& name, const double& value);
		bool setValueByName(const std::string& name, const bool& value, const bool& flipBool = true);
		bool setValueByName(const std::string& name, const unsigned int& value);
		bool setValueByName(const std::string& name, std::string value);
		void calculateOverallProbability();
	};

// This JBrain class assumes a growth paradigm:
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
		// outputs are disembodied dendrites or neurons with no axons:
		std::vector<JDendrite> m_outputDendrites;
		std::vector<JNeuron> m_outputNeurons;
		std::vector<JAxon> m_inputAxons;

		// Dendrites:
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
		bool m_useOutputNeurons;

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

		// Output calculations ignore inputs from input axons:
		bool m_brainOutputsIgnoreEnvironmentInputs;

		// How many brain time steps happen between when the environment values
		// are provided to the brain and output is read from the brain.
		unsigned int m_brainProcessingStepsBetweenInputAndOutput;
		unsigned int m_brainOutputsToAverageTogether;

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
		// The updater for dendrites attached to the output neurons:
		CGP::JBrainCGPIndividual* m_CGPOutputDendriteUpdater;
		CGP::JBrainCGPIndividual* m_CGPAxonUpdater;
		CGP::JBrainCGPIndividual* m_CGPNeuronUpdater;
		CGP::JBrainCGPIndividual* m_CGPChemicalUpdater;
		bool areEqual(CGP::JBrainCGPIndividual* lhs, CGP::JBrainCGPIndividual* rhs);

		// The inputs for each CGP program:
		std::vector<CGP::CGP_INPUT> m_dendriteInputs;
		std::vector<CGP::CGP_OUTPUT> m_dendriteOutputs;
		std::vector<CGP::CGP_INPUT> m_outputDendriteInputs;
		std::vector<CGP::CGP_OUTPUT> m_outputDendriteOutputs;
		std::vector<CGP::CGP_INPUT> m_axonInputs;
		std::vector<CGP::CGP_OUTPUT> m_axonOutputs;
		std::vector<CGP::CGP_INPUT> m_neuronInputs;
		std::vector<CGP::CGP_OUTPUT> m_neuronOutputs;

		// The number of CGP nodes to allocate to each program:
		unsigned int m_dendriteProgramNodes;
		unsigned int m_outputDendriteProgramNodes;
		unsigned int m_axonProgramNodes;
		unsigned int m_neuronProgramNodes;

		// Use our class variables to create each of the singular updaters.
		void createDendriteUpdater();
		void createOutputDendriteUpdater();
		void createAxonUpdater();
		void createNeuronUpdater();
		void createChemicalUpdater();
		void createAllSeparateUpdaters(); // Call the 4 other creation functions.

		// Update frequency:
		CGP::UPDATE_EVENT m_updateEvent;
		unsigned int m_updateFrequency;

		// Equations available to the JBrainCGPIndividuals:
		std::vector<std::string> m_functionStringList;
		std::vector<std::function<double(double, double, double)> > m_functionList;

		// All non-output neurons:
		std::vector<JNeuron> m_neurons;

		// Records of the choices this brain made and choices made by
		// the sage (if used). This is used to calculate the percentage of time
		// we make the "right" choice:
		std::vector<int> m_sageChoices;
		std::vector<int> m_brainChoices;
		std::vector<double> m_mostRecentBrainOutput;
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
		float getDendriteInput(JDendrite& dendrite, const bool& ignoreEnvironmentInputs=false);

		// Check if neuron fires:
		bool getIfNeuronFires(JNeuron& neuron);
		float calculateInternalNeuronValue(JNeuron& neuron, const bool& ignoreEnvironmentInputs=false);

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

		// Dendrite version requires the parent neuron. They alse use the
		// neuron number so that recent output values can be referenced:
		void applyCGP(JDendrite& dendrite, const unsigned int& parentNeuronNum, const bool& outputNeuron);
		std::vector<double> getCGPInputs(const JDendrite& dendrite, const unsigned int& parentNeuronNum, const bool& outputNeuron);
		void applyCGPOutputs(JDendrite& dendrite, const std::vector<double>& cgpOutputs,
			const unsigned int& parentNeuronNum, const bool& outputNeuron);

		void applyCGP(JAxon& axon, const JNeuron& parentNeuron);
		std::vector<double> getCGPInputs(const JAxon& axon, const JNeuron& parentNeuron);
		void applyCGPOutputs(JAxon& axon, const std::vector<double>& cgpOutputs,
			const JNeuron& parentNeuron);

		// With neuron health values all updated, kill/duplicate those that have
		// crossed those thresholds.
		void duplicateAndKillNeurons();
		JNeuron createDuplicateNeuron(JNeuron& neuron);
		
		// CreateNewNeuron:
		// -1: Random dendrites/axon counts:
		JNeuron createNewNeuron(const float& x, const float& y, const float& z,
			const int& dendriteCount = -1, const int& axonCount = -1);

		// Gather status values:
		unsigned int getDendriteCount();
		unsigned int getAxonCount();
		float getAverageNeuronHealth();
		
	public:
		inline std::string getName() { return m_name; }
		inline std::string getParentName() { return m_parentName; }
		inline bool getUseOutputNeurons() { return m_useOutputNeurons; }
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

		void setNeuronsFromStaticJson(json& neuronJson, const bool& outputNeurons);

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
			const bool& needInitializeUpdatersAndConnections = true);
		
		JBrain(std::string yamlFilename);
		void writeSelfToJson(json& j);
		static JBrain* getBrainFromJson(json& j);
		void writeSelfHumanReadable(std::ostream& out);
	};

} // End namespace JBrain