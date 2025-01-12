#include "pch.h"
#include "JBrain.h"
#include <random>
#include <set>
#include <cmath>

#include <chrono>

namespace JBrain
{
	JBrain_Snap::~JBrain_Snap()
	{
		closeCSVOutputFile();

		// Clean up all of our allocated neurons:
		for (auto elem : m_allNeurons)
		{
			if (elem != nullptr)
			{
				delete elem;
				elem = nullptr;
			}
		}
	}

	JBrain_Snap::JBrain_Snap(
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
		const double& dendriteWeightTickDownAmount,
		const double& dendriteCorrectWeightChange,
		const double& dendriteIncorrectWeightChange,
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
		const double& outputNegative_CreatePureProcessingNeuron,
		const double& noOutput_IncreaseInputDendriteWeight,
		const double& noOutput_AddProcessingNeuronDendrite,
		const double& noOutput_IncreaseProcessingNeuronDendriteWeight,
		const double& noOutput_AddOutputNeuronDendrite,
		const double& noOutput_IncreaseOutputNeuronDendriteWeight,
		const double& noOutput_CreateProcessingNeuron,
		const double& noOutput_CreatePureProcessingNeuron,
		const bool& usePassthroughInputNeurons,
		const bool& useHDCMode,
		const unsigned int& hdcMinimumDeleteDistance,
		const CGP::HDC_LEARN_MODE& hdcLearnMode,
		ObservationProcessor* observationProcessor)
		:
	m_name(name),
		m_parentName(parentName),
		m_staticOverallProbability(overallProbability),
		m_dynamicProbabilityUsage(dynamicProbabilityUsage),
		m_dynamicProbabilityMultiplier(dynamicProbabilityMultiplier),
		m_mostRecentScorePercent(1.0),
		m_neuronAccumulateDuration(neuronAccumulateDuration),
		m_neuronResetOnFiring(neuronResetOnFiring),  // Not used yet
		m_neuronResetAfterOutput(neuronResetAfterOutput),
		m_neuronFireThreshold(neuronFireThreshold),
		m_neuronMaximumAge(neuronMaximumAge),
		m_brainProcessingStepsAllowed(brainProcessingStepsAllowed),
		m_initialInputNeuronCount(initialInputNeuronCount),
		m_initialProcessingNeuronCount(initialProcessingNeuronCount),
		m_maximumProcessingNeuronCount(maximumProcessingNeuronCount),
	  m_maximumInputNeuronsToInputRatio(maximumInputNeuronToInputsRatio),
		m_dendriteWeightChange(dendriteWeightChange),
		m_dendriteMinimumWeight(dendriteMinimumWeight),
		m_dendriteMaximumWeight(dendriteMaximumWeight),
		m_dendriteStartingWeight(dendriteStartingWeight),
		m_dendriteWeightTickDownAmount(dendriteWeightTickDownAmount),
		m_dendriteCorrectWeightChange(dendriteCorrectWeightChange),
		m_dendriteIncorrectWeightChange(dendriteIncorrectWeightChange),
		m_dendriteMinCountPerNeuron(dendriteMinCountPerNeuron),
		m_dendriteMaxCountPerNeuron(dendriteMaxCountPerNeuron),
		m_dendriteStartCountPerNeuron(dendriteStartCountPerNeuron),
		m_baseProcessingNeuronCount(baseProcessingNeuronCount),
		m_actionSize(actionSize),		
		m_stepCreateNeuronChance(stepCreateNeuronChance),
		m_stepCreateNeuron_BaseCountRatioMultiplier(stepCreateNeuron_BaseCountRatioMultiplier),
		m_stepCreateInputNeuronChance(stepCreateInputNeuronChance),
	  m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier(stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier),
		m_stepDestroyNeuronChance(stepDestroyNeuronChance),
		m_stepDestroyNeuron_CountBaseRatioMultiplier(stepDestroyNeuron_CountBaseRatioMultiplier),
		m_stepDestroyInputNeuronChance(stepDestroyInputNeuronChance),
		m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier(stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier),
		m_destroyNeuron_FavorFewerConnections(destroyNeuron_FavorFewerConnections),
		m_destroyNeuron_FavorYoungerNeurons(destroyNeuron_FavorYoungerNeurons),
		m_runCreateNeuronChance(runCreateNeuronChance),
		m_runCreateNeuron_BaseCountRatioMultiplier(runCreateNeuron_BaseCountRatioMultiplier),
		m_runCreateInputNeuronChance(runCreateInputNeuronChance),
		m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier(runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier),
		m_runDestroyNeuronChance(runDestroyNeuronChance),
		m_runDestroyNeuron_CountBaseRatioMultiplier(runDestroyNeuron_CountBaseRatioMultiplier),
		m_runDestroyInputNeuronChance(runDestroyInputNeuronChance),
		m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier(runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier),
		m_outputPositive_CascadeProbability(outputPositive_CascadeProbability),
		m_outputPositive_InSequence_IncreaseDendriteWeight(outputPositive_InSequence_IncreaseDendriteWeight),
		m_outputPositive_NoConnection_InSequence_CreateConnection(outputPositive_NoConnection_InSequence_CreateConnection),
		m_outputPositive_YesFire_UnusedInput_DecreaseWeight(outputPositive_YesFire_UnusedInput_DecreaseWeight),
		m_outputPositive_YesFire_UnusedInput_BreakConnection(outputPositive_YesFire_UnusedInput_BreakConnection),
		m_outputNegative_CascadeProbability(outputNegative_CascadeProbability),
		m_outputNegative_InSequence_DecreaseDendriteWeight(outputNegative_InSequence_DecreaseDendriteWeight),
		m_outputNegative_InSequence_BreakConnection(outputNegative_InSequence_BreakConnection),
		m_outputNegative_CreatePureProcessingNeuron(outputNegative_CreatePureProcessingNeuron),
		m_noOutput_IncreaseInputDendriteWeight(noOutput_IncreaseInputDendriteWeight),
	  m_noOutput_AddProcessingNeuronDendrite(noOutput_AddProcessingNeuronDendrite),
		m_noOutput_IncreaseProcessingNeuronDendriteWeight(noOutput_IncreaseProcessingNeuronDendriteWeight),
		m_noOutput_AddOutputNeuronDendrite(noOutput_AddOutputNeuronDendrite),
		m_noOutput_IncreaseOutputNeuronDendriteWeight(noOutput_IncreaseOutputNeuronDendriteWeight),
		m_noOutput_CreateProcessingNeuron(noOutput_CreateProcessingNeuron),
		m_noOutput_CreatePureProcessingNeuron(noOutput_CreatePureProcessingNeuron),
		m_observationProcessor(observationProcessor),
		m_usePassthroughInputNeurons(usePassthroughInputNeurons),
		m_useHDCMode(useHDCMode),
		m_hdcMinimumDeleteDistance(hdcMinimumDeleteDistance),
		m_hdcLearnMode(hdcLearnMode),
		m_outputCSV(nullptr)		
	{
		// correct output neuron is set with each process-input call:
		m_correctOutputNeuron = -1;
		m_correctOutputAction = -1;

		// Observation size is set by the observation processor:
		m_observationSize = m_observationProcessor->getExpectedOutputSize();

		// Create all starting neurons:
		createAllStartingNeurons(m_initialInputNeuronCount, m_initialProcessingNeuronCount);

		// Logging set to start:
		resetAllLoggingValues();

		// Get probabilities set:
		calculateOverallProbability();
	}
	
	double JBrain_Snap::getChance_CorrectGotInput_IncreaseWeight()
	{
		return m_outputPositive_InSequence_IncreaseDendriteWeight * m_overallProbability;
	}
	
	double JBrain_Snap::getChance_CorrectGotInput_CreateConnection()
	{
		return m_outputPositive_NoConnection_InSequence_CreateConnection * m_overallProbability;
	}

	double JBrain_Snap::getChance_YesFired_UnusedInput_DecreaseWeight()
	{
		return m_outputPositive_YesFire_UnusedInput_DecreaseWeight * m_overallProbability;
	}

	double JBrain_Snap::getChance_YesFired_UnusedInput_BreakConnection()
	{
		return m_outputPositive_YesFire_UnusedInput_BreakConnection * m_overallProbability;
	}

	double JBrain_Snap::getChance_WrongGotInput_DecreaseWeight()
	{
		return m_outputNegative_InSequence_DecreaseDendriteWeight * m_overallProbability;
	}

	double JBrain_Snap::getChance_WrongGotInput_BreakConnection()
	{
		return m_outputNegative_InSequence_BreakConnection * m_overallProbability;
	}

	double JBrain_Snap::getChance_WrongOutput_CreatePureProcessingNeuron()
	{
		return m_overallProbability * m_outputNegative_CreatePureProcessingNeuron;
	}

	double JBrain_Snap::getChance_Step_CreateProcessingNeuron()
	{
		double baseOverCount = static_cast<double>(m_baseProcessingNeuronCount) /
			fmax(static_cast<double>(m_processingNeurons.size()), 0.1); // fmax to prevent div-by-zero problems

		double fullProbability = m_overallProbability * (m_stepCreateNeuronChance +
			(baseOverCount * m_stepCreateNeuron_BaseCountRatioMultiplier));

		return fullProbability;
	}

	double JBrain_Snap::getChance_Step_DestroyProcessingNeuron()
	{
		double countOverBase = static_cast<double>(m_processingNeurons.size()) / static_cast<double>(m_baseProcessingNeuronCount);
		double fullProbability = m_overallProbability * 
			(m_stepDestroyNeuronChance + (countOverBase * m_stepDestroyNeuron_CountBaseRatioMultiplier));

		return fullProbability;
	}

	double JBrain_Snap::getChance_Step_CreateInputNeuron()
	{
		double obsOverCount = static_cast<double>(m_observationSize) / 
			fmax(static_cast<double>(m_inputNeurons.size()), 1.0);
		
		return m_overallProbability * (m_stepCreateInputNeuronChance +
			(obsOverCount * m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier));
	}

	double JBrain_Snap::getChance_Step_DestroyInputNeuron()
	{
		double countOverObs = static_cast<double>(m_inputNeurons.size()) / static_cast<double>(m_observationSize);

		return m_overallProbability * (m_stepDestroyInputNeuronChance +
			(countOverObs * m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier));	
	}

	double JBrain_Snap::getChance_Run_CreateProcessingNeuron()
	{
		double baseOverCount = static_cast<double>(m_baseProcessingNeuronCount) /
			fmax(static_cast<double>(m_processingNeurons.size()), 0.1); // fmax to prevent div-by-zero problems

		double fullProbability =  m_overallProbability * (m_runCreateNeuronChance +
			(baseOverCount * m_runCreateNeuron_BaseCountRatioMultiplier));

		return fullProbability;
	}

	double JBrain_Snap::getChance_Run_DestroyProcessingNeuron()
	{
		double countOverBase = static_cast<double>(m_processingNeurons.size()) / static_cast<double>(m_baseProcessingNeuronCount);
		double fullProbability = m_overallProbability * 
			(m_runDestroyNeuronChance + (countOverBase * m_runDestroyNeuron_CountBaseRatioMultiplier));

		return fullProbability;
	}

	double JBrain_Snap::getChance_Run_CreateInputNeuron()
	{
		double obsOverCount = static_cast<double>(m_observationSize) /
			fmax(static_cast<double>(m_inputNeurons.size()), 1.0);

		return m_overallProbability * (m_runCreateInputNeuronChance +
			(obsOverCount * m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier));
	}

	double JBrain_Snap::getChance_Run_DestroyInputNeuron()
	{
		double countOverObs = static_cast<double>(m_inputNeurons.size()) / static_cast<double>(m_observationSize);

		return m_overallProbability * (m_runDestroyInputNeuronChance +
			(countOverObs * m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier));
	}

	double JBrain_Snap::getChance_NoOut_IncreaseInputDendriteWeight()
	{
		return m_overallProbability * m_noOutput_IncreaseInputDendriteWeight;
	}

	double JBrain_Snap::getChance_NoOut_AddProcessingNeuronDendrite()
	{
		return m_overallProbability * m_noOutput_AddProcessingNeuronDendrite;
	}

	double JBrain_Snap::getChance_NoOut_IncreaseProcessingNeuronDendriteWeight()
	{
		return m_overallProbability * m_noOutput_IncreaseProcessingNeuronDendriteWeight;
	}

	double JBrain_Snap::getChance_NoOut_AddOutputNeuronDendrite()
	{
		return m_overallProbability * m_noOutput_AddOutputNeuronDendrite;
	}
	
	double JBrain_Snap::getChance_NoOut_IncreaseOutputNeuronDendriteWeight()
	{
		return m_overallProbability * m_noOutput_IncreaseOutputNeuronDendriteWeight;
	}
	
	double JBrain_Snap::getChance_NoOut_CreateProcessingNeuron()
	{
		return m_overallProbability * m_noOutput_CreateProcessingNeuron;
	}

	double JBrain_Snap::getChance_NoOut_CreatePureProcessingNeuron()
	{
		return m_overallProbability * m_noOutput_CreatePureProcessingNeuron;
	}
	
	void JBrain_Snap::calculateOverallProbability()
	{
		double dynamicProbability = m_mostRecentScorePercent * m_dynamicProbabilityMultiplier;
		// Use our most recent score to calculate the dynamic overall probability:
		switch (m_dynamicProbabilityUsage)
		{
		case CGP::DYNAMIC_PROBABILITY::UNDEFINED:
			std::cout << "Error: Undefined dynamic probability." << std::endl;
			[[fallthrough]];  // Let the compiler know I did this on purpose

		case CGP::DYNAMIC_PROBABILITY::UNUSED:  // Don't apply dynamic probability at all.
			m_overallProbability = m_staticOverallProbability;
			break;

		case CGP::DYNAMIC_PROBABILITY::ADD:
			m_overallProbability = dynamicProbability + m_staticOverallProbability;
			break;

		case CGP::DYNAMIC_PROBABILITY::MULTIPLY:
			m_overallProbability = dynamicProbability * m_staticOverallProbability;
			break;

		case CGP::DYNAMIC_PROBABILITY::SOLO:
			m_overallProbability = dynamicProbability;
			break;
		}
	}

	bool JBrain_Snap::getInsideMaximumInputNeuronToInputsRatio()
	{
		return (static_cast<double>(m_inputNeurons.size()) / static_cast<double>(m_observationSize)) 
			< m_maximumInputNeuronsToInputRatio;
	}

	bool JBrain_Snap::getInsideMaximumProcessingNeuronsCount()
	{
		return static_cast<unsigned int>(m_processingNeurons.size()) < m_maximumProcessingNeuronCount;
	}

	void JBrain_Snap::ensureAllInputsUsed()
	{
		// This function shouldn't be called if we are using pass-through input neurons:
		if (m_usePassthroughInputNeurons || m_useHDCMode)
			return;

		// If there is an input used 0 times, keep creating input neurons:
		std::vector<unsigned int> inputsUsed = getUsedInputsCount();
		auto iter = std::find(inputsUsed.begin(), inputsUsed.end(), 0);
				
		// While we can find a '0' in the input used count, keep creating input neurons:
		while (iter != inputsUsed.end() && getInsideMaximumInputNeuronToInputsRatio())
		{
			doCreateInputNeuron();
			inputsUsed = getUsedInputsCount();
			iter = std::find(inputsUsed.begin(), inputsUsed.end(), 0);			
		}
	}

	void JBrain_Snap::createAllStartingNeurons(const unsigned int& inputCount, const unsigned int& processingCount)
	{
		// Because they make dendrite connection choices based on what is available, create
		// input neurons, then processing, then output:
		if (m_usePassthroughInputNeurons)
			createAllPassthroughInputNeurons();
		else if (!m_useHDCMode)
		{
			for (unsigned int i = 0; i < inputCount; ++i)
				doCreateInputNeuron();

			ensureAllInputsUsed();  // May need to create more input neurons
		
			for (unsigned int i = 0; i < processingCount; ++i)
				doCreateProcessingNeuron();

			for (unsigned int i = 0; i < m_actionSize; ++i)
				doCreateOutputNeuron();
		}
		
		if (m_useHDCMode)
			doCreateHDCOutputNeurons();
	}

	bool JBrain_Snap::getEventHappened(double probability)
	{
		// Random device and distribution don't need to be
		// recreated every time:
		static std::random_device rd;
		static std::mt19937_64 eng(rd());
		static std::uniform_real_distribution<double> distr(0.0, 1.0);

		if (distr(eng) <= probability)
			return true;
		else
			return false;
	}

	unsigned int JBrain_Snap::getNeuronCount()
	{
		return static_cast<unsigned int>(std::count_if(m_allNeurons.begin(), m_allNeurons.end(),
			[](JNeuron_Snap* x) {return x != nullptr; }));
	}

	std::vector<double> JBrain_Snap::processInput(const std::vector<double>& inputs, const std::vector<double>& sageChoice)
	{
		// Set the correct neuron to fire. Output neurons should be in the order that
		// we output:
		if (!m_useHDCMode)
		{
			m_correctOutputAction = std::round(sageChoice[0]);
			m_correctOutputNeuron = m_outputNeurons[m_correctOutputAction]->m_neuronNumber;
		}

		// Get our processed input vector:
		m_mostRecentBrainInputs = m_observationProcessor->processInput(inputs);
		
		if (m_useHDCMode)
		{
			m_hdcSimplifiedInput = m_observationProcessor->getSeparatedInputs_simplified();
			m_hdcCorrectOutputNeurons = m_observationProcessor->getSeparatedOutputs_simplified(sageChoice);
		}
		
		bool gotResponse = false;
		std::vector<double> brainOutputs {};

		unsigned int DEBUG_RUN_COUNT = 0;
		unsigned int MAX_DRC = 10;
		while (!gotResponse && DEBUG_RUN_COUNT < MAX_DRC)
		{
			gotResponse = runProcessingSteps(brainOutputs);
			if (!gotResponse)
			{
				++m_noOutputHappenedCount;
				handleNoOutputEvents();
			}

			++DEBUG_RUN_COUNT;
		}

		if (DEBUG_RUN_COUNT >= MAX_DRC)
		{
			// std::cout << "DEBUG_RUN_COUNT >= " << MAX_DRC << ". Probably didn't get a good brain output." << std::endl;
		}

		// HDC Needs to record when the right/wrong output neuron fires separately
		if (m_useHDCMode)
		{
			for (unsigned int idx = 0; idx < m_hdcCorrectOutputNeurons.size(); ++idx)
			{
				// Different way to check
				/*if (brainOutputs[m_correctOutputAction] > 0.5)
					++m_correctNeuronFiredCount;
				else
					++m_wrongNeuronFiredCount;*/
			}
		}
				
		// Even if we didn't fire, we've reached the end of handling a single input. Process output events:
		handleOutputEvents();

		return brainOutputs;
	}

	void JBrain_Snap::handleHDCOutputUpdate(const std::vector<double>& brainOutputs)
	{
		static bool allOrNoneRemoval = true;

		// If we got the correct output, check to see if we need to update:
		if (m_hdcLearnMode == CGP::HDC_LEARN_MODE::NONE)
		{
				return;
		}
		
		// If we got the answer perfectly correct, there is no need to update, even in full-mode:
		if (m_hdcCorrectOutputNeurons == m_hdcSimplifiedOutput)
			return;

		// At least one output is wrong, create a new processing neuron:
		unsigned int neuNum = static_cast<unsigned int>(m_allNeurons.size());
		JNeuron_Snap* tempNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::PROCESSING,
			neuNum, 1.0, m_actionSize);

		// Add all of the inputs:
		for (auto inVal : m_hdcSimplifiedInput)
		{
			tempNeuron->m_inputNeurons.push_back(inVal);
			tempNeuron->m_inputWeights.push_back(1.0);
		}

		// Initial starting fire value is a distance of zero:
		tempNeuron->m_fireValue = 0;

		// Add it to our output calculations:
		// Add it to each of our output bucket sets:
		for (unsigned int outNum = 0; outNum < m_hdcCorrectOutputNeurons.size(); ++outNum)
		{
			m_hdcOutputNeurons[outNum][m_hdcCorrectOutputNeurons[outNum]]->m_inputNeurons.push_back(neuNum);
			m_hdcOutputNeurons[outNum][m_hdcCorrectOutputNeurons[outNum]]->m_inputWeights.push_back(1.0);
		}

		// Add it to our lists of neurons:
		++m_processingNeuronCreatedCount;
		m_allNeurons.push_back(tempNeuron);
		m_processingNeurons.push_back(tempNeuron);

		// If the wrong neuron had values that were too close, remove them:
		unsigned int procNeuNum;
		unsigned int sumDistance = 0;
		JNeuron_Snap* nPtr = nullptr;
		JNeuron_Snap* procNeu;
		for (unsigned int idx = 0; idx < m_hdcOutputNeurons.size(); ++idx)
		{
			nPtr = m_hdcOutputNeurons[idx][m_hdcSimplifiedOutput[idx]];
			sumDistance += nPtr->m_fireValue;
		}

		// With multiple outputs, we only deal with replacing exactly equal processing neurons.
		if (sumDistance <= 0)
		{
			// There should be exactly 1 value in nPtr's output list:
			procNeuNum = nPtr->m_outputNeurons[0];
			doDestroyProcessingNeuron(procNeuNum);
		}
	}

	void JBrain_Snap::handleHDCNoOutputUpdate()
	{
		// No output usually indicates no training. Warn the user if we aren't training:
		if (m_hdcLearnMode == CGP::HDC_LEARN_MODE::NONE)
		{
			std::cout << "ERROR: No output in HDC mode, but training set to None." << std::endl;
			return;
		}

		// No output in HDC mode means we need to add an HDC processing neuron:
		unsigned int neuNum = static_cast<unsigned int>(m_allNeurons.size());
		JNeuron_Snap* tempNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::PROCESSING,
			neuNum, 1.0, m_actionSize);

		// Add all of the inputs:
		for (auto inVal : m_hdcSimplifiedInput)
		{
			tempNeuron->m_inputNeurons.push_back(inVal);
			tempNeuron->m_inputWeights.push_back(1.0);
		}

		// Initial starting fire value is a distance of zero:
		tempNeuron->m_fireValue = 0;

		// Add it to each of our output bucket sets:
		for (unsigned int outNum = 0; outNum < m_hdcCorrectOutputNeurons.size(); ++outNum)
		{
			m_hdcOutputNeurons[outNum][m_hdcCorrectOutputNeurons[outNum]]->m_inputNeurons.push_back(neuNum);
			m_hdcOutputNeurons[outNum][m_hdcCorrectOutputNeurons[outNum]]->m_inputWeights.push_back(1.0);
		}

		// Add it to our lists of neurons:
		++m_processingNeuronCreatedCount;
		m_allNeurons.push_back(tempNeuron);
		m_processingNeurons.push_back(tempNeuron);
	}

	unsigned int JBrain_Snap::readSingleHDCOutput(const std::vector<JNeuron_Snap*>& outBuckets)
	{
		std::vector<double> retVal(outBuckets.size(), 0.0);

		// Find the minimum index:
		unsigned int minIdx = 0;
		double minVal = outBuckets[0]->m_fireValue;
		for (unsigned int idx = 1; idx < outBuckets.size(); ++idx)
		{
			if (outBuckets[idx]->m_fireValue < minVal)
			{
				minIdx = idx;
				minVal = outBuckets[idx]->m_fireValue;
			}
		}

		return minIdx;
	}

	std::vector<double> JBrain_Snap::readHDCBrainOutput()
	{
		m_hdcSimplifiedOutput.clear();
		for (unsigned int outNum = 0; outNum < m_hdcOutputNeurons.size(); ++outNum)
		{
			m_hdcSimplifiedOutput.push_back(readSingleHDCOutput(m_hdcOutputNeurons[outNum]));
		}

		return m_observationProcessor->processOutput(m_hdcSimplifiedOutput);
	}

	std::vector<double> JBrain_Snap::readBrainOutput(const unsigned int& stepNumber)
	{
		if (m_useHDCMode)
			return readHDCBrainOutput();

		std::vector<double> retVal {};
		
		// Brain outputs should be in the order of the neurons:
		for (auto& nPtr : m_outputNeurons)
		{
			// Search when the output neuron fired, if it fired on the provided step number,
			// add a 1.0 to the output, otherwise add a 0.0:
			auto iter = std::find(nPtr->m_fireSteps.begin(), nPtr->m_fireSteps.end(), stepNumber);
			if (iter == nPtr->m_fireSteps.end())
				retVal.push_back(0.0);
			else
				retVal.push_back(1.0);
		}

		return retVal;
	}

	bool JBrain_Snap::fireAllProcessingAndOutputNeurons(const unsigned int& stepNumber)
	{
		if (stepNumber < 1)
			std::cout << "ERROR: Running fireAllProcessingNeurons on step " << stepNumber << std::endl;

		// Don't just use '-' to prevent the unsigned integers from rolling over:
		unsigned int minStep = 0;
		unsigned int maxStep = stepNumber - 1;  // Must be at least step 1 for processing neurons to fire.
		if (stepNumber >= m_neuronAccumulateDuration)
			minStep = stepNumber - m_neuronAccumulateDuration;
		
		bool outputNeuronFired = false;
		// For every processing neuron:
		for (unsigned int neuNum = 0; neuNum < m_allNeurons.size(); ++neuNum)
		{
			// Null ptr or an input neuron, skip:
			if (m_allNeurons[neuNum] == nullptr || m_allNeurons[neuNum]->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
				continue;
			else // This function should handle cascading good/bad output events:
			{
				// True, check if it was an output neuron, if so, need to return true:
				if (setIfNonInputNeuronFired(neuNum, stepNumber, minStep, maxStep))
				{
					if (m_allNeurons[neuNum]->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
					{
						outputNeuronFired = true;
						++m_outputNeuronFiredCount;
						if (!m_useHDCMode)  // Handle these events in non-hdc mode:
						{
							// Right vs wrong neuron. It is rare, but possible that they both fire
							// at the same time and all weights will increase and decrease.
							if (neuNum == m_correctOutputNeuron)
							{
								++m_correctNeuronFiredCount;
								handleCorrectOutputNeuronFiredEvent(neuNum, stepNumber);
							}
							else
							{
								++m_wrongNeuronFiredCount;
								handleWrongOutputNeuronFiredEvent();
							}
						}
					}
					else
					{
						++m_processingNeuronFiredCount;
					}
				}
			}
		}
		return outputNeuronFired;
	}

	void JBrain_Snap::resetAllNeuronFires()
	{
		// Mark all neurons as having never fired:
		for (auto& nPtr : m_allNeurons)
		{
			if (nPtr != nullptr)
			{
				nPtr->m_fireSteps.clear();
				if (m_useHDCMode)
					nPtr->m_fireValue = -1.0;
			}
		}
	}

	void JBrain_Snap::uniformDendriteWeightChange(const double& change)
	{
		// HDC Handles weight changes differently:
		if (m_useHDCMode)
			return;
		
		// For all neurons, change the weight of all dendrites:
		for (auto nPtr : m_allNeurons)
		{
			if (nPtr != nullptr)
			{
				for (auto wIter = nPtr->m_inputWeights.begin(); wIter != nPtr->m_inputWeights.end(); ++wIter)
					*wIter += change;
			}
		}
	}

	bool JBrain_Snap::runProcessingSteps(std::vector<double>& brainOutputs)
	{
		// Reset all neurons:
		resetAllNeuronFires();

		// Fire the input neurons, the only neurons that can fire on step 0:
		for (auto& nPtr : m_inputNeurons)
		{
			setIfInputNeuronFired(nPtr->m_neuronNumber, 0);
		}

		// Fire all neurons until we run out of time or an output neuron fires:
		bool outputNeuronFired = false;
		unsigned int stepNum;
		for (stepNum = 1; stepNum < m_brainProcessingStepsAllowed; ++stepNum)
		{
			// Handle all of the chance happenings for each step:
			handleStepEvents();

			// If at least 1 output neuron fired, we're done.
			if (outputNeuronFired = fireAllProcessingAndOutputNeurons(stepNum))
				break;
		}

		// Read the output neurons in any case:
		brainOutputs = readBrainOutput(stepNum);

		// Before resetting the neuron-fires (that we need for decision-making), update the
		// hdc-specific neurons:
		if (m_useHDCMode && outputNeuronFired)
			handleHDCOutputUpdate(brainOutputs);

		if (outputNeuronFired && m_neuronResetAfterOutput)
			resetAllNeuronFires();

		// Return true if we produced an output:
		return outputNeuronFired;
	}

	bool JBrain_Snap::setIfInputNeuronFired(const unsigned int& neuronNumber, const int& currentStepNumber)
	{
		double sum = 0.0;
		bool neuronFired = false;		
		auto nPtr = m_allNeurons[neuronNumber];

		// Passthrough input neurons just check to see if their given input is set or not:
		if (m_usePassthroughInputNeurons)
		{
			unsigned int inputNumber = nPtr->m_inputNeurons[0];
			if (m_mostRecentBrainInputs[inputNumber] > 0.1)  // !!! Only works with binary for now !!!
				neuronFired = true;
		}
		else  // Standard input neurons
		{
			// Input neurons use their "m_inputNeurons" vector to refer to brain inputs from external sources:
			for (unsigned int i = 0; i < nPtr->m_inputNeurons.size(); ++i)
			{
				sum += m_mostRecentBrainInputs[nPtr->m_inputNeurons[i]] * nPtr->m_inputWeights[i];
			}

			if (sum >= nPtr->m_fireThreshold)
			{
				neuronFired = true;
			}
		}
		
		if (neuronFired)
		{
			// Mark that it fired when this was the correct output action:
			++nPtr->m_firedExpectedOutputCounts[m_correctOutputAction];
			nPtr->m_fireSteps.push_back(currentStepNumber);
			++m_inputNeuronFiredCount;
		}
		
		return neuronFired;
	}

	bool JBrain_Snap::setIfHDCProcessingNeuronFired(JNeuron_Snap* neuron)
	{
		int totalDist = 0;
		assert(neuron->m_inputNeurons.size() == m_hdcSimplifiedInput.size());

		for (unsigned int idx = 0; idx < neuron->m_inputNeurons.size(); ++idx)
		{
			totalDist += abs(static_cast<int>(neuron->m_inputNeurons[idx]) -
				static_cast<int>(m_hdcSimplifiedInput[idx]));
		}

		neuron->m_fireValue = static_cast<double>(totalDist);
		++m_processingNeuronFiredCount;
		return true;
	}

	bool JBrain_Snap::setIfHDCOutputNeuronFired(JNeuron_Snap* neuron, const unsigned int& stepNumber)
	{
		// We only fire beyond the first non-input-neuron step(1).
		if (stepNumber < 2)
			return false;

		// Use rounding to deal with potential off-by-a-tiny-amount double errors.
		// Output neurons don't use their m_outputNeurons variable, so we will use it to track
		// the neurons that matched our minimum value:
		neuron->m_fireValue = static_cast<double>(std::numeric_limits<int>::max() - 1);  // Start high, but within int bounds
		neuron->m_outputNeurons.clear();
		JNeuron_Snap* nPtr;
		bool retVal = false;  // Need at least one input to return true

		for (auto inNeuNum : neuron->m_inputNeurons)
		{
			nPtr = m_allNeurons[inNeuNum];
			if (nPtr != nullptr)
			{
				retVal = true;

				// With HDC, lower difference is better rather than higher accumulated value:
				if (static_cast<int>(std::round(nPtr->m_fireValue)) < static_cast<int>(std::round(neuron->m_fireValue)))
				{
					neuron->m_fireValue = nPtr->m_fireValue;
					neuron->m_outputNeurons.clear();
					neuron->m_outputNeurons.push_back(inNeuNum);
				}
				// Found an equal value, add it to the list of evidence:
				else if (static_cast<int>(std::round(nPtr->m_fireValue)) == static_cast<int>(std::round(neuron->m_fireValue)))
				{
					neuron->m_outputNeurons.push_back(inNeuNum);
				}
				//else // Do nothing.
			}
		}

		return retVal;
	}

	bool JBrain_Snap::setIfNonInputNeuronFired(const unsigned int& neuronNumber, const int& currentStepNumber,
		const unsigned int& minAccumulateStep, const unsigned int& maxAccumulateStep)
	{
		auto nPtr = m_allNeurons[neuronNumber];
		bool retVal = false;

		if (nPtr == nullptr || nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
			return retVal;
		
		// In HDC mode, neurons fire-checks work differently:
		if (m_useHDCMode)
		{
			if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::PROCESSING)
				return setIfHDCProcessingNeuronFired(nPtr);
			else if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
				return setIfHDCOutputNeuronFired(nPtr, currentStepNumber);
			else
			{
				std::cout << "setIfNonInputNeuronFired bad neuron type: "
					<< CGP::JneuronSnapTypeToString(nPtr->m_type) << std::endl;
				return false;
			}
		}

		double sum = 0.0;
		// Check every step where the previous neuron firing matters:
		for (unsigned int checkStep = minAccumulateStep; checkStep <= maxAccumulateStep; ++checkStep)
		{
			// Check every neuron used as an input to this neuron:
			for (unsigned int nIdx = 0; nIdx < nPtr->m_inputNeurons.size(); ++nIdx)
			{
				// It is possible the neuron has been removed, skip if it has:
				if (m_allNeurons[nPtr->m_inputNeurons[nIdx]] == nullptr)
					continue;

				if (m_allNeurons[nPtr->m_inputNeurons[nIdx]]->getFiredOnStepNum(checkStep))
				{
					// 1.0 is the only fire value, for now:
					sum += (1.0 * nPtr->m_inputWeights[nIdx]);

					// If this is an output neuron, need to apply some changes to the brain:
					if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
					{
						if (neuronNumber == m_correctOutputNeuron)
						{
							++m_correctOutNeuGotInputCalledCount;
							handleCorrectOutputNeuronGotInputEvent(neuronNumber, currentStepNumber);
						}
						else
						{
							++m_wrongOutNeuGotInputCalledCount;
							handleWrongOutputNeuronGotInputEvent(neuronNumber, currentStepNumber);
						}
					}
				}
			}
		}

		if (sum >= nPtr->m_fireThreshold)
		{
			retVal = true;
			nPtr->m_fireSteps.push_back(currentStepNumber);

			// Record that it fired when this was the correct output:
			++nPtr->m_firedExpectedOutputCounts[m_correctOutputAction];
		}

		return retVal;
	}

	void JBrain_Snap::handleWrongOutputNeuronGotInputEvent(const unsigned int& neuron,
		const unsigned int& stepNumber, double decreaseWeightChance, double breakConnectionChance)
	{
		// This function is recursive. It should be activated on an output neuron, it will modify that
		// neuron, then call itself on all input neurons to which it should cascade with modified
		// chances of activation. The recursion ends on step 0 or when modifying an input neuron.
		unsigned int idx;
		auto nPtr = m_allNeurons[neuron];
		if (nPtr == nullptr)
		{
			std::cout << "handleWrongOutputNeuronGotInputEvent function got a null neuron #" << neuron << std::endl;
			return;
		}

		else if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
		{
			// For now, assume inputs only fire on step 0. If we're here AND it is step 0, possibly
			// create dendrites and/or change weights:
			if (stepNumber == 0)
			{
				if (getEventHappened(decreaseWeightChance))
					doHandleInputNeuronDecreaseWeights(neuron);

				if (getEventHappened(breakConnectionChance))
					doHandleInputNeuronBreakConnection(neuron);
			}
		}

		else if (stepNumber > 0)
		{
			// Output neuron, we need to calculate the chances to pass recursively to other function calls:
			if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
			{
				decreaseWeightChance = getChance_WrongGotInput_DecreaseWeight();
				breakConnectionChance = getChance_WrongGotInput_BreakConnection();
			}

			std::vector<JNeuron_Snap*> firedInSeq = getAllNeuronsFiredOnStep(stepNumber - 1);
			for (auto inPtr : firedInSeq)
			{
				// If we have a connection, consider decreasing our weight to it:
				auto iter1 = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), inPtr->m_neuronNumber);
				if (iter1 != nPtr->m_inputNeurons.end())
				{
					idx = static_cast<unsigned int>(iter1 - nPtr->m_inputNeurons.begin());
					// If this weight should decrease, do so, but don't let it go below the minimum:
					if (getEventHappened(decreaseWeightChance))
						nPtr->m_inputWeights[idx] -= m_dendriteWeightChange;

					// If we should break the connection, do so:
					if (getEventHappened(breakConnectionChance))
					{
						// Only drop if it isn't violating the rules:
						if (nPtr->m_inputNeurons.size() > m_dendriteMinCountPerNeuron)
							nPtr->dropInput(inPtr->m_neuronNumber);
					}

					// Cascade this possibility to the previous neuron in the sequence:
					handleWrongOutputNeuronGotInputEvent(inPtr->m_neuronNumber, stepNumber - 1,
						decreaseWeightChance * m_outputNegative_CascadeProbability,
						breakConnectionChance * m_outputNegative_CascadeProbability);
				} // End if we had a connection.				
			} // End for all neurons that fired a step before this neuron.
		} // End else if stepNumber > 0
	}

	std::vector<unsigned int> JBrain_Snap::getUsedInputsCount()
	{
		// Start with zeros:
		std::vector<unsigned int> retVal(m_observationSize, 0);

		for (auto nPtr : m_inputNeurons)
		{
			for (auto inputNum : nPtr->m_inputNeurons)
			{
				++retVal[inputNum];
			}
		}

		return retVal;
	}

	std::vector<unsigned int> JBrain_Snap::getOutputCountVector(const std::vector<JNeuron_Snap*> checkVec)
	{
		std::vector<unsigned int> retVal {};

		for (auto nPtr : checkVec)
		{
			retVal.push_back(static_cast<unsigned int>(nPtr->m_outputNeurons.size()));
		}

		return retVal;
	}

	void JBrain_Snap::handleStepEvents()
	{
		// Age all neurons:
		for (auto& nPtr : m_allNeurons)
		{
			if (nPtr != nullptr)			
				nPtr->m_age = std::min(nPtr->m_age + 1, m_neuronMaximumAge);			
		}
		
		// HDC has different update stages for processing/input neurons:
		if (m_useHDCMode)
			return;

		// For each event, check probability and activate if it occurs:		
		if (getEventHappened(getChance_Step_CreateProcessingNeuron()))
			if (getInsideMaximumProcessingNeuronsCount())
				doCreateProcessingNeuron();

		if (getEventHappened(getChance_Step_DestroyProcessingNeuron()))
			doDestroyProcessingNeuron();

		if (getEventHappened(getChance_Step_CreateInputNeuron()))
			if (getInsideMaximumInputNeuronToInputsRatio())
				doCreateInputNeuron();

		if (getEventHappened(getChance_Step_DestroyInputNeuron()))
			doDestroyInputNeuron();
	}

	void JBrain_Snap::handleEndOfRunEvents()
	{
		// HDC has different update stages for processing/input neurons:
		if (m_useHDCMode)
			return;

		// For each event, check probability and activate if it occurs:
		if (getEventHappened(getChance_Run_CreateProcessingNeuron()))
			if (getInsideMaximumProcessingNeuronsCount())
				doCreateProcessingNeuron();

		if (getEventHappened(getChance_Run_DestroyProcessingNeuron()))
			doDestroyProcessingNeuron();

		if (getEventHappened(getChance_Run_CreateInputNeuron()))
			if (getInsideMaximumInputNeuronToInputsRatio())
				doCreateInputNeuron();

		if (getEventHappened(getChance_Run_DestroyInputNeuron()))
			doDestroyInputNeuron();
	}

	// The brain is finished producing a single output from an input:
	void JBrain_Snap::handleOutputEvents()
	{
		// HDC output events get handled BEFORE we reset all neuron fire states:
		if (m_useHDCMode)
			return;

		// All weights tick down a bit:
		uniformDendriteWeightChange(-m_dendriteWeightTickDownAmount);

		// Delete all dendrites with weights below the minimum:
		deleteAllDendritesWithBelowMinimumWeights();

		// Delete all neurons that now have too few dendrites:
		deleteAllNeuronsWithTooFewDendrites();
	}

	// The brain produced no output, some heavy-handed changes:
	void JBrain_Snap::handleNoOutputEvents()
	{
		if (m_useHDCMode)
		{
			handleHDCNoOutputUpdate();
			return;
		}
		
		// Create a new processing neurons BEFORE creating dendrites on output neurons:
		if (getEventHappened(getChance_NoOut_CreateProcessingNeuron()))
			if (getInsideMaximumProcessingNeuronsCount())
				doCreateProcessingNeuron();

		// Create a new pure processing neuron:
		if (getEventHappened(getChance_NoOut_CreatePureProcessingNeuron()))
			if (getInsideMaximumProcessingNeuronsCount())
				doCreatePureProcessingNeuron();

		// Add dendrites to output neurons:
		for (auto nPtr : m_outputNeurons)
		{
			if (getEventHappened(getChance_NoOut_AddOutputNeuronDendrite()))
				doAddOutputNeuronDendrite(nPtr);
		}

		// Add dendrites to processing neurons:
		for (auto nPtr : m_processingNeurons)
		{
			if (getEventHappened(getChance_NoOut_AddProcessingNeuronDendrite()))
				doAddProcessingNeuronDendrite(nPtr);
		}

		// Increase weight of input neuron dendrites:
		if (!m_usePassthroughInputNeurons)
		{
			for (auto nPtr : m_inputNeurons)
			{
				for (unsigned int idx = 0; idx < nPtr->m_inputWeights.size(); ++idx)
				{
					if (getEventHappened(getChance_NoOut_IncreaseInputDendriteWeight()))
						nPtr->m_inputWeights[idx] = fmin(m_dendriteMaximumWeight, nPtr->m_inputWeights[idx] + m_dendriteWeightChange);
				}
			}
		}

		for (auto nPtr : m_outputNeurons)
		{
			for (unsigned int idx = 0; idx < nPtr->m_inputWeights.size(); ++idx)
			{
				if (getEventHappened(getChance_NoOut_IncreaseOutputNeuronDendriteWeight()))
					nPtr->m_inputWeights[idx] = fmin(m_dendriteMaximumWeight, nPtr->m_inputWeights[idx] + m_dendriteWeightChange);
			}
		}		
	}

	void JBrain_Snap::doAddOutputNeuronDendrite(JNeuron_Snap* outputNeuron)
	{
		static std::mt19937_64 gen(std::random_device{}());
		static double minimumPurity = 0.0;  // Maybe a parameter, eventually?
		
		// Values used to make sure we don't end up with all zeros in chances:		
		static double addedPurity = 0.0001;

		// Determine the output we should be working towards:
		unsigned int expectedOutput = 0;
		bool foundNeuron = false;

		for (unsigned int i = 0; i < m_outputNeurons.size(); ++i)
		{
			if (m_outputNeurons[i] == outputNeuron)
			{
				foundNeuron = true;
				expectedOutput = i;
				break;
			}
		}

		if (!foundNeuron)
		{
			std::cout << "doAddOutputNeuronDendrite called with neuron not in output neurons." << std::endl;
		}

		if (outputNeuron->m_inputNeurons.size() >= m_dendriteMaxCountPerNeuron)
			return;

		// Get a list of all processing and input neurons to choose from by gathering all acceptable
		// neurons from the inputs and processing neuron lists.
		double tempPurity;
		std::vector<JNeuron_Snap*> possibleInputNeurons;
		std::vector<double> chances;

		for (auto nPtr : m_processingNeurons)
		{
			tempPurity = nPtr->getFirePurity(expectedOutput);
			if (tempPurity >= minimumPurity)
			{
				possibleInputNeurons.push_back(nPtr);
				chances.push_back(tempPurity + addedPurity);
			}
		}

		// Remove all those we already have a connection to:
		JNeuron_Snap* findPtr;
		unsigned int idx;
		for (auto val : outputNeuron->m_inputNeurons)
		{
			findPtr = m_allNeurons[val];
			auto findItr = std::find(possibleInputNeurons.begin(), possibleInputNeurons.end(), findPtr);
			if (findItr != possibleInputNeurons.end())
			{
				idx = findItr - possibleInputNeurons.begin();
				possibleInputNeurons.erase(findItr);
				chances.erase(chances.begin() + idx);
			}
		}

		// If there are no more in the list, return:
		if (possibleInputNeurons.size() < 1)
		{
			// std::cout << "doAddOutputNeuronDendrite: No input neurons available." << std::endl;
			return;
		}

		// Otherwise, choose from the available neurons according to their purity:
		std::discrete_distribution<std::size_t> dist{ chances.begin(), chances.end() };
		unsigned int neuronNumber = possibleInputNeurons[dist(gen)]->m_neuronNumber;

		// Add our connection to it and its connection to us:
		outputNeuron->m_inputNeurons.push_back(neuronNumber);
		outputNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
		m_allNeurons[neuronNumber]->m_outputNeurons.push_back(outputNeuron->m_neuronNumber);
	}

	void JBrain_Snap::doAddProcessingNeuronDendrite(JNeuron_Snap* procNeuron)
	{
		static std::mt19937_64 gen(std::random_device{}());

		if (procNeuron->m_inputNeurons.size() >= m_dendriteMaxCountPerNeuron)
			return;

		// Get a list of all processing and input neurons to choose from:
		std::vector<JNeuron_Snap*> possibleInputNeurons{};
		std::copy(m_inputNeurons.begin(), m_inputNeurons.end(), std::back_inserter(possibleInputNeurons));
		std::copy(m_processingNeurons.begin(), m_processingNeurons.end(), std::back_inserter(possibleInputNeurons));

		// Remove all those we already have a connection to:
		for (auto val : procNeuron->m_inputNeurons)
		{
			auto findPtr = m_allNeurons[val];
			auto findItr = std::find(possibleInputNeurons.begin(), possibleInputNeurons.end(), findPtr);
			if (findItr != possibleInputNeurons.end())
				possibleInputNeurons.erase(findItr);
		}

		// If there are no more in the list, return:
		if (possibleInputNeurons.size() < 1)
			return;

		// Otherwise, create choose from the available neurons:
		std::uniform_int_distribution<> dist(0, static_cast<int>(possibleInputNeurons.size()) - 1);
		unsigned int idx = static_cast<unsigned int>(dist(gen));

		// Add our connection to it and its connection to us:
		procNeuron->m_inputNeurons.push_back(possibleInputNeurons[idx]->m_neuronNumber);
		procNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
		possibleInputNeurons[idx]->m_outputNeurons.push_back(procNeuron->m_neuronNumber);
	}

	void JBrain_Snap::getFullTrialStatistics(unsigned int& noOutputEvents, unsigned int& goodOutputs,
		unsigned int& badOutputs, unsigned int& procNeuronCreated, unsigned int& procNeuronDestroyed,
		unsigned int& inputNeuronCreated, unsigned int& inputNeuronDestroyed)
	{
		noOutputEvents = m_noOutputHappenedCount;
		goodOutputs = m_correctNeuronFiredCount;
		badOutputs = m_wrongNeuronFiredCount;
		procNeuronCreated = m_processingNeuronCreatedCount;
		procNeuronDestroyed = m_processingNeuronDestroyedCount;
		inputNeuronCreated = m_inputNeuronCreatedCount;
		inputNeuronDestroyed = m_inputNeuronDestroyedCount;
	}

	void JBrain_Snap::processEndOfTrial(double reward, const double& minReward, double maxReward)
	{
		// Deal with potentially negative scores and set a baseline of 0 by adding (0 - min) to all:
		double scoreAdd = 0 - minReward;		
		maxReward += scoreAdd;
		reward += scoreAdd;

		// Now, this should calculate the percentage of max reward we got. Update our dynamic
		// probability before checking on other event probabilities:		
		m_mostRecentScorePercent = fmin(reward / maxReward, 1.0); // Min in case use gave bad min/max.
		calculateOverallProbability();
		
		// Handle the events:
		handleEndOfRunEvents();

		// Write line to CSV:
		writeLineToCSVOutputFile(reward);
		resetAllLoggingValues();		
	}

	void JBrain_Snap::handleWrongOutputNeuronFiredEvent()
	{
		doChangeUsedDendriteConnectionWeights_AllNeurons(m_dendriteIncorrectWeightChange);

		// Create a pure processing neuron when we make the wrong output:
		if (getEventHappened(getChance_WrongOutput_CreatePureProcessingNeuron()))
			if (getInsideMaximumProcessingNeuronsCount())
				doCreatePureProcessingNeuron();
	}

	void JBrain_Snap::handleCorrectOutputNeuronFiredEvent(const unsigned int& neuron, const unsigned int& stepNumber)
	{
		auto nPtr = m_allNeurons[neuron];
		if (nPtr == nullptr)
		{
			std::cout << "handleCorrectOutputNeuronFiredEvent function got a null neuron #" << neuron << std::endl;
			return;
		}

		// Fired on step zero shouldn't happen, but just to be careful:
		if (stepNumber == 0)
			return;

		// Do the blanket weight change for all neurons that contributed to this outcome:
		doChangeUsedDendriteConnectionWeights_AllNeurons(m_dendriteCorrectWeightChange);

		// Output neuron fired event doesn't cascade.  It just has a chance to break connections
		// with neurons that were unused.
		std::vector<JNeuron_Snap*> firedNeurons = getAllNeuronsFiredOnStep(stepNumber - 1);

		// Go through our inputs. If they aren't in the fired-neuron list, make changes:
		if (!m_usePassthroughInputNeurons)
		{
			for (auto inNeuNum : nPtr->m_inputNeurons)
			{
				auto nIter = std::find_if(firedNeurons.begin(), firedNeurons.end(),
					[inNeuNum](JNeuron_Snap* x) {return x->m_neuronNumber == inNeuNum; });

				// Didn't find it in the fired neurons:
				if (nIter == firedNeurons.end())
				{
					if (getEventHappened(getChance_YesFired_UnusedInput_DecreaseWeight()))
						doDecreaseDendriteWeight(neuron, inNeuNum);

					if (getEventHappened(getChance_YesFired_UnusedInput_BreakConnection()))
						doDropDendriteConnection(neuron, inNeuNum);
				}
			}
		}
	}

	void JBrain_Snap::deleteAllNeuronsWithTooFewDendrites()
	{
		JNeuron_Snap* nPtr;
		// Sort through all neurons, if it isn't used enough, delete it:
		for (int i = static_cast<int>(m_allNeurons.size()) - 1; i >= 0; --i)
		{
			nPtr = m_allNeurons[i];
			if (nPtr != nullptr && nPtr->m_inputNeurons.size() < m_dendriteMinCountPerNeuron)
			{
				if (!m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
					doDestroyInputNeuron(static_cast<const int>(nPtr->m_neuronNumber));
				else if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::PROCESSING)
					doDestroyProcessingNeuron(static_cast<const int>(nPtr->m_neuronNumber));
				else if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)  // Should not happen
					doAddOutputNeuronDendrite(nPtr);
				else
				{
					std::cout << "Bad type for neuron: " << nPtr->m_neuronNumber
						<< ": " << CGP::JneuronSnapTypeToString(nPtr->m_type) << std::endl;
				}				
			} // end if not null and too few dendrites
		} // end for all neurons
	}

	// This function is called assuming that the specified neuron fired on step stepNum and we want to
	// increase the weight of all dendrites that may have contributed to that. So, for all of the
	// accumulation steps, if the input neuron fired, we increase the weight to it.
	void JBrain_Snap::doChangeUsedProcessingDendriteConnectionWeights(const unsigned int& neuronNumber, const unsigned int& stepNum, const double& weightValue)
	{
		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doChangeUsedProcessingDendriteConnectionWeights called with a null neuron number." << std::endl;
			return;
		}

		// Get the steps we care about:
		std::vector<unsigned int> steps {};
		
		// Need to calculate in signed integers, but then get an unsigned back to avoid rolling
		// the unsigned value back to a large number if going below zero:
		unsigned int minStep = static_cast<unsigned int>(
			std::min(static_cast<int>(stepNum) - static_cast<int>(m_neuronAccumulateDuration), 0) );
		unsigned int maxStep = stepNum - 1;
		for (unsigned int step = minStep; step <= maxStep; ++step)
			steps.push_back(step);
		
		// Get which neurons fired during those steps:
		std::vector<bool> fired = getConnectedNeuronsFiredOnStep(steps, nPtr->m_inputNeurons);

		// The bool vector should line up with our input vector:
		assert(fired.size() == nPtr->m_inputNeurons.size());

		// For each that fired, change the weight:
		for (unsigned int i = 0; i < fired.size(); ++i)
		{
			if (fired[i])
			{
				nPtr->m_inputWeights[i] += weightValue;
			}
		}
	}

	void JBrain_Snap::doChangeUsedProcessingDendriteConnectionWeights(const unsigned int& neuronNumber, const double& weightValue)
	{
		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doChangeUsedProcessingDendriteConnectionWeights called with a null neuron number." << std::endl;
			return;
		}

		// For every step that this neuron fired, run the update-weight function:
		for (auto step : nPtr->m_fireSteps)
			doChangeUsedProcessingDendriteConnectionWeights(neuronNumber, step, weightValue);
	}

	void JBrain_Snap::doChangeUsedInputDendriteConnectionWeights(const unsigned int& neuronNumber, const double& weightValue)
	{
		// Not used with passthrough input neurons:
		if (m_usePassthroughInputNeurons)
			return;

		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doChangeUsedInputDendriteConnectionWeights called with a null neuron number." << std::endl;
			return;
		}

		// Input neurons use m_inputNeurons to refer to system inputs. For each of those that fired,
		// change our weight to it:
		for (unsigned int i = 0; i < nPtr->m_inputNeurons.size(); ++i)
		{
			if (m_mostRecentBrainInputs[ nPtr->m_inputNeurons[i] ] > 0.0)
			{
				nPtr->m_inputWeights[i] += weightValue;
			}
		}		
	}

	void JBrain_Snap::doChangeUsedDendriteConnectionWeights_AllNeurons(const double& weightValue)
	{
		for (auto nPtr : m_allNeurons)
		{
			if (nPtr == nullptr)
				continue;
			else if (!m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
				doChangeUsedInputDendriteConnectionWeights(nPtr->m_neuronNumber, weightValue);
			else if (nPtr->m_type != CGP::JNEURON_SNAP_TYPE::INPUT)  // Works for processing or output neurons:
				doChangeUsedProcessingDendriteConnectionWeights(nPtr->m_neuronNumber, weightValue);
		}
	}

	void JBrain_Snap::handleCorrectOutputNeuronGotInputEvent(const unsigned int& neuron,
		const unsigned int& stepNumber, double increaseWeightChance, double createConnectionChance)
	{
		// This function is recursive. It should be activated on an output neuron, it will modify that
		// neuron, then call itself on all input neurons to which it should cascade with modified
		// chances of activation. The recursion ends on step 0 or when modifying an input neuron.
		unsigned int idx;
		auto nPtr = m_allNeurons[neuron];
		if (nPtr == nullptr)
		{
			std::cout << "handleCorrectOutputNeuronGotInputEvent function got a null neuron #" << neuron << std::endl;
			return;
		}

		else if (!m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
		{
			// For now, assume inputs only fire on step 0. If we're here AND it is step 0, possibly
			// create dendrites and/or change weights:
			if (stepNumber == 0)
			{
				if (getEventHappened(increaseWeightChance))
					doHandleInputNeuronIncreaseWeights(neuron);

				if (getEventHappened(createConnectionChance))
					doHandleInputNeuronCreateConnection(neuron);
			}
		}

		else if (stepNumber > 0)
		{
			// Output neuron, we need to calculate the chances to pass recursively to other function calls:
			if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
			{
				increaseWeightChance = getChance_CorrectGotInput_IncreaseWeight();
				createConnectionChance = getChance_CorrectGotInput_CreateConnection();
			}

			std::vector<JNeuron_Snap*> firedInSeq = getAllNeuronsFiredOnStep(stepNumber - 1);
			for (auto inPtr : firedInSeq)
			{
				// If we have a connection, consider increasing our weight to it:
				auto iter1 = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), inPtr->m_neuronNumber);
				if (iter1 != nPtr->m_inputNeurons.end())
				{
					idx = static_cast<unsigned int>(iter1 - nPtr->m_inputNeurons.begin());
					// If this weight should increase, do so, but don't let it go over the maximum:
					if (getEventHappened(increaseWeightChance))
						nPtr->m_inputWeights[idx] = fmin(nPtr->m_inputWeights[idx] + m_dendriteWeightChange, m_dendriteMaximumWeight);

					// Cascade this possibility to the previous neuron in the sequence:
					handleCorrectOutputNeuronGotInputEvent(inPtr->m_neuronNumber, stepNumber - 1,
						increaseWeightChance * m_outputPositive_CascadeProbability,
						createConnectionChance * m_outputPositive_CascadeProbability);
				} // End if we already had a connection.
				else
				{
					// If we do not have a connection, consider adding one, if it isn't ourselves:
					if (inPtr != nPtr && getEventHappened(createConnectionChance))
					{
						// Only if we haven't hit the maximum dendrite count:
						if (nPtr->m_inputNeurons.size() < m_dendriteMaxCountPerNeuron)
						{
							nPtr->m_inputNeurons.push_back(inPtr->m_neuronNumber);
							nPtr->m_inputWeights.push_back(m_dendriteStartingWeight);

							// Only if we created a connection to this neuron do we want to cascade:
							handleCorrectOutputNeuronGotInputEvent(inPtr->m_neuronNumber,
								stepNumber - 1, increaseWeightChance * m_outputPositive_CascadeProbability,
								createConnectionChance * m_outputPositive_CascadeProbability);
						}
					}						
				} // End if we did not have a connection to the neuron
			} // End for all neurons that fired a step before this neuron.
		} // End else if stepNumber > 0
	}

	void JBrain_Snap::doHandleInputNeuronIncreaseWeights(const unsigned int& neuron)
	{
		if (m_usePassthroughInputNeurons)
			return;

		auto nPtr = m_allNeurons[neuron];
		
		// If we take input from it AND it fired, increase the weight:
		for (unsigned int i = 0; i < nPtr->m_inputNeurons.size(); ++i)
		{			
			if (m_mostRecentBrainInputs[nPtr->m_inputNeurons[i]] > 0.0)
			{
				nPtr->m_inputWeights[i] = fmin(nPtr->m_inputWeights[i] + m_dendriteWeightChange, m_dendriteMaximumWeight);
			}				
		}
	}

	void JBrain_Snap::doHandleInputNeuronCreateConnection(const unsigned int& neuron)
	{
		if (m_usePassthroughInputNeurons)
			return;

		auto nPtr = m_allNeurons[neuron];

		// Don't exceed the maximum:
		if (nPtr->m_inputNeurons.size() >= m_dendriteMaxCountPerNeuron)
			return;

		// Create a list of all inputs that fired and for which we don't have a connection:
		std::vector<unsigned int> firedNoConnectionIdx {};
		for (unsigned int idx = 0; idx < m_mostRecentBrainInputs.size(); ++idx)
		{
			if (m_mostRecentBrainInputs[idx] > 0.0)
			{
				auto iter = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), idx);
				if (iter == nPtr->m_inputNeurons.end()) // Couldn't find it.
				{
					// Make sure we aren't connection to ourselves?? 
					firedNoConnectionIdx.push_back(idx);
				}
			}
		}

		// Allocate the random classes only once:
		static std::random_device rd;
		static std::mt19937_64 gen(rd());

		unsigned int selection = 0; // Assume only a single connection found
		if (firedNoConnectionIdx.size() == 0)
			return;
		else if (firedNoConnectionIdx.size() > 1) // More than 1 found, select randomly:
		{
			std::uniform_int_distribution<> dis(0, static_cast<int>(firedNoConnectionIdx.size()) - 1);
			selection = static_cast<unsigned int>(dis(gen));
		}

		nPtr->m_inputNeurons.push_back(firedNoConnectionIdx[selection]);
		nPtr->m_inputWeights.push_back(m_dendriteStartingWeight);
	}
	
	void JBrain_Snap::doHandleInputNeuronDecreaseWeights(const unsigned int& neuron)
	{
		if (m_usePassthroughInputNeurons)
			return;

		auto nPtr = m_allNeurons[neuron];

		// If we take input from it AND it fired, decrease the weight:
		for (unsigned int i = 0; i < nPtr->m_inputNeurons.size(); ++i)
		{
			if (m_mostRecentBrainInputs[nPtr->m_inputNeurons[i]] > 0.0)
			{
				nPtr->m_inputWeights[i] -= m_dendriteWeightChange;
			}
		}
	}

	void JBrain_Snap::initializeCSVOutputFile(std::string dataDirectory)
	{
		closeCSVOutputFile();

		// Get our folder / file name:
		std::stringstream sstream;
		sstream << dataDirectory << m_name << ".csv";
		std::string filename = sstream.str();
		m_outputCSV = new std::ofstream(filename.c_str());

		if (m_outputCSV == nullptr || !m_outputCSV->good())
		{
			std::cout << "Failed to open " << filename << std::endl;
			closeCSVOutputFile();
		}		

		// The full list of columns:
		*m_outputCSV << "finalScore,\
inputNeuronCount,\
outputNeuronCount,\
processingNeuronCount,\
allNeuronsListSize,\
rightNeuronFires,\
wrongNeuronFires,\
neuronZeroFires,\
neuronOneFires,\
inputNeuronFires,\
processingNeuronFires,\
outputNeuronFires,\
inputNeuronsCreated,\
processingNeuronsCreated,\
inputNeuronsDestroyed,\
processingNeuronsDestroyed,\
correctOutputNeuronGotInputCount,\
wrongOutputNeuronGotInputCount,\
noOutputHappened" << std::endl;
	}

	void JBrain_Snap::writeLineToCSVOutputFile(const double& score)
	{
		if (m_outputCSV == nullptr)
		{
			std::cout << "Trying to write to null output file pointer." << std::endl;
			return;
		}

		*m_outputCSV << score << ","
			<< m_inputNeurons.size() << ","
			<< m_outputNeurons.size() << ","
			<< m_processingNeurons.size() << ","
			<< m_allNeurons.size() << ","
			<< m_correctNeuronFiredCount << ","
			<< m_wrongNeuronFiredCount << ","
			<< m_brainOutputZeroFiredCount << ","
			<< m_brainOutputOneFiredCount << ","
			<< m_inputNeuronFiredCount << ","
			<< m_processingNeuronFiredCount << ","
			<< m_outputNeuronFiredCount << ","
			<< m_inputNeuronCreatedCount << ","
			<< m_processingNeuronCreatedCount << ","
			<< m_inputNeuronDestroyedCount << ","
			<< m_processingNeuronDestroyedCount << ","
			<< m_correctOutNeuGotInputCalledCount << ","
			<< m_wrongOutNeuGotInputCalledCount << ","
			<< m_noOutputHappenedCount << std::endl;
	}

	void JBrain_Snap::resetAllLoggingValues()
	{
		m_correctNeuronFiredCount = 0;
		m_wrongNeuronFiredCount = 0;
		m_brainOutputZeroFiredCount = 0;
		m_brainOutputOneFiredCount = 0;
		m_inputNeuronFiredCount = 0;
		m_processingNeuronFiredCount = 0;
		m_outputNeuronFiredCount = 0;
		m_inputNeuronCreatedCount = 0;
		m_inputNeuronDestroyedCount = 0;
		m_processingNeuronCreatedCount = 0;
		m_processingNeuronDestroyedCount = 0;
		m_correctOutNeuGotInputCalledCount = 0;
		m_wrongOutNeuGotInputCalledCount = 0;
		m_noOutputHappenedCount = 0;
	}

	void JBrain_Snap::closeCSVOutputFile()
	{
		if (m_outputCSV != nullptr)
		{
			m_outputCSV->close();
			delete m_outputCSV;
		}

		m_outputCSV = nullptr;
	}

	void JBrain_Snap::doHandleInputNeuronBreakConnection(const unsigned int& neuron)
	{
		if (m_usePassthroughInputNeurons)
			return;

		auto nPtr = m_allNeurons[neuron];
		
		// Don't destroy if it lowers our count too far:
		if (nPtr->m_inputNeurons.size() <= m_dendriteMinCountPerNeuron)
			return;

		// Create a list of all inputs that fired that we have inputs for:
		std::vector<unsigned int> firedConnectionIdx{};
		for (unsigned int idx = 0; idx < m_mostRecentBrainInputs.size(); ++idx)
		{
			if (m_mostRecentBrainInputs[idx] > 0.0)
			{
				auto iter = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), idx);
				if (iter != nPtr->m_inputNeurons.end()) // Did find it:
					firedConnectionIdx.push_back(static_cast<unsigned int>(iter - nPtr->m_inputNeurons.begin()));
			}
		}

		// Allocate the random classes only once:
		static std::random_device rd;
		static std::mt19937_64 gen(rd());

		unsigned int selection = 0; // Assume only a single connection found
		if (firedConnectionIdx.size() == 0)
			return;
		else if (firedConnectionIdx.size() > 1) // More than 1 found, select randomly:
		{
			std::uniform_int_distribution<> dis(0, static_cast<int>(firedConnectionIdx.size()) - 1);
			selection = static_cast<unsigned int>(dis(gen));
		}

		nPtr->m_inputNeurons.erase(nPtr->m_inputNeurons.begin() + selection);
		nPtr->m_inputWeights.erase(nPtr->m_inputWeights.begin() + selection);
	}

	std::vector<JNeuron_Snap*> JBrain_Snap::getAllNeuronsFiredOnStep(const unsigned int& stepNumber)
	{
		std::vector<JNeuron_Snap*> retVal {};
		for (JNeuron_Snap* nPtr : m_allNeurons)
		{
			if (nPtr != nullptr && nPtr->getFiredOnStepNum(stepNumber))
				retVal.push_back(nPtr);
		}
		return retVal;
	}

	std::vector<bool> JBrain_Snap::getConnectedNeuronsFiredOnStep(const std::vector<unsigned int>& steps,
		const std::vector<unsigned int>& neuronNumsToCheck)
	{
		std::vector<bool> retVal {};
		bool singleNeuronFire;
		JNeuron_Snap* nPtr;
		for (auto nNum : neuronNumsToCheck)
		{
			nPtr = m_allNeurons[nNum];
			singleNeuronFire = false;
			
			// Skip null neurons:
			if (nPtr != nullptr)
			{
				// For every neuron, check every step:
				for (auto step : steps)
				{
					if (nPtr->getFiredOnStepNum(step))
					{
						// Don't need to know how many times it fired, just true/false.
						singleNeuronFire = true;
						break;
					}
				} // End for all steps
			} // End if neuron isn't null
			
			retVal.push_back(singleNeuronFire);
		} // End for all neurons to check
		
		return retVal;
	}

	void JBrain_Snap::doCreateHDCOutputNeurons()
	{
		// Get how many output buckets are allowed for each output:
		std::vector<unsigned int> outBucketSizes = m_observationProcessor->getActionBucketSizes();

		// Create all the output neurons:
		m_hdcOutputNeurons.clear();
		std::vector<JNeuron_Snap*> tempOutNeurons;
		JNeuron_Snap* tempNeuron;
		unsigned int neuronNumber;

		// For every output, create a number of output neurons equal to the number of buckets
		// available for that output:
		for (unsigned int outNum = 0; outNum < outBucketSizes.size(); ++outNum)
		{
			tempOutNeurons.clear();
			for (unsigned int bucketNum = 0; bucketNum < outBucketSizes[outNum]; ++bucketNum)
			{
				neuronNumber = static_cast<unsigned int>(m_allNeurons.size());
				tempNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::OUTPUT, neuronNumber, 1.0, outBucketSizes.size());
				tempOutNeurons.push_back(tempNeuron);
				m_allNeurons.push_back(tempNeuron);
			}
			m_hdcOutputNeurons.push_back(tempOutNeurons);
		}
	}

	void JBrain_Snap::doCreateOutputNeuron()
	{ 
		static std::mt19937_64 gen(std::random_device{}());

		// Get a list of all processing neurons that we can modify:
		std::vector<JNeuron_Snap*> possibleInputNeurons{};
		std::copy(m_processingNeurons.begin(), m_processingNeurons.end(), std::back_inserter(possibleInputNeurons));

		unsigned int nNumber = static_cast<unsigned int>(m_allNeurons.size());
		JNeuron_Snap* newNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::OUTPUT, nNumber,
			m_neuronFireThreshold, m_actionSize);

		// If there aren't enough inputs to satisfy dendrite start count, do our best:
		if (m_dendriteStartCountPerNeuron > possibleInputNeurons.size())
		{
			for (auto nPtr : possibleInputNeurons)
			{
				newNeuron->m_inputNeurons.push_back(nPtr->m_neuronNumber);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
				nPtr->m_outputNeurons.push_back(nNumber);
			}
		}

		// There are at least enough. Choose randomly; no favoring implemented, yet:
		else
		{
			while (static_cast<unsigned int>(newNeuron->m_inputNeurons.size()) < m_dendriteStartCountPerNeuron)
			{
				// New distribution each loop as the input list size changes:
				std::uniform_int_distribution<> dist(0, static_cast<int>(possibleInputNeurons.size()) - 1);
				unsigned int idx = static_cast<unsigned int>(dist(gen));

				// Add our connection to it and its connection to us:
				newNeuron->m_inputNeurons.push_back(possibleInputNeurons[idx]->m_neuronNumber);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
				possibleInputNeurons[idx]->m_outputNeurons.push_back(nNumber);

				// Remove it from the possibilities:
				possibleInputNeurons.erase(possibleInputNeurons.begin() + idx);
			}
		}

		m_allNeurons.push_back(newNeuron);
		m_outputNeurons.push_back(newNeuron);
	}

	void JBrain_Snap::doDecreaseDendriteWeight(const unsigned int& neuronNumber, const unsigned int& inputNeuronNumber)
	{
		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doDecreaseDendriteWeight called with nullptr neuron number." << std::endl;
			return;
		}

		// No changes to input neurons if using passthrough input neurons:
		if (m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
			return;

		// Find the input and decrease the weight:
		auto inIter = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), inputNeuronNumber);		
		if (inIter != nPtr->m_inputNeurons.end())
		{
			unsigned int idx = static_cast<unsigned int>(inIter - nPtr->m_inputNeurons.begin());
			
			// Subtract off the amount dendrite weights change:
			nPtr->m_inputWeights[idx] -= m_dendriteWeightChange;			
		}
	}

	void JBrain_Snap::deleteAllDendritesWithBelowMinimumWeights()
	{
		// Check all neurons:
		for (auto nPtr : m_allNeurons)
		{
			if (nPtr == nullptr)
				continue;

			// Move from the back of the list to the front so we can remove as we go:
			for (int i = static_cast<int>(nPtr->m_inputWeights.size()) - 1; i >= 0; --i)
			{
				if (nPtr->m_inputWeights[i] < m_dendriteMinimumWeight)
				{
					doDeleteDendrite(nPtr->m_neuronNumber, static_cast<unsigned int>(i));

					// Output neuorns can't be deleted, so if they have too few dendrites, create more:
					if (nPtr->m_type == CGP::JNEURON_SNAP_TYPE::OUTPUT)
					{
						if (nPtr->m_inputNeurons.size() < m_dendriteMinCountPerNeuron)
							doAddOutputNeuronDendrite(nPtr);
					}
				}
			}
		}
	}

	void JBrain_Snap::doDeleteDendrite(const unsigned int& neuronNumber, const unsigned int& dendriteNumber)
	{
		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doDeleteDendrite called with nullptr neuron number." << std::endl;
			return;
		}

		// No changes to input neurons if using passthrough input neurons:
		if (m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
			return;

		// If this wasn't an input neuron, erase from the output neuron we pulled from:
		if (nPtr->m_type != CGP::JNEURON_SNAP_TYPE::INPUT) 
		{
			JNeuron_Snap* iPtr = m_allNeurons[nPtr->m_inputNeurons[dendriteNumber]];
			if (iPtr != nullptr)
			{
				auto outIter = std::find(iPtr->m_outputNeurons.begin(), iPtr->m_outputNeurons.end(), neuronNumber);
				// This should always be true, but double-check anyway:
				if (outIter != iPtr->m_outputNeurons.end())
					iPtr->m_outputNeurons.erase(outIter);
			}
			else // BUG
			{
				std::cout << "BUG: Neuron " << neuronNumber << " (" << CGP::JneuronSnapTypeToString(nPtr->m_type)
					<< ") had a pointer to a null neuron in its inputs. [" 
					<< nPtr->m_inputNeurons[dendriteNumber] << "]." << std::endl;
			}
		}

		// Delete it from inputs:
		nPtr->m_inputNeurons.erase(nPtr->m_inputNeurons.begin() + dendriteNumber);
		nPtr->m_inputWeights.erase(nPtr->m_inputWeights.begin() + dendriteNumber);
	}

	void JBrain_Snap::doDropDendriteConnection(const unsigned int& neuronNumber, const unsigned int& inputNeuronNumber)
	{
		JNeuron_Snap* nPtr = m_allNeurons[neuronNumber];
		if (nPtr == nullptr)
		{
			std::cout << "doDropDendriteConnection called with nullptr neuron number." << std::endl;
			return;
		}

		// No changes to input neurons if using passthrough input neurons:
		if (m_usePassthroughInputNeurons && nPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
			return;

		// Find that neuron number in our input dendrites:
		auto inIter = std::find(nPtr->m_inputNeurons.begin(), nPtr->m_inputNeurons.end(), inputNeuronNumber);
		
		// If we find it, eliminate it:		
		if (inIter != nPtr->m_inputNeurons.end())
		{
			unsigned int idx = static_cast<unsigned int>(inIter - nPtr->m_inputNeurons.begin());
			doDeleteDendrite(neuronNumber, idx);
		}
	}

	void JBrain_Snap::doCreatePureProcessingNeuron()
	{
		static double minimumPurity = 0.0;  // Maybe a parameter, eventually?
		
		// Values used to make sure we don't end up with all zeros in chances:
		static double addedPurity = 0.0001;

		static std::mt19937_64 gen(std::random_device{}());		
		++m_processingNeuronCreatedCount;

		// Get a list of all processing and input neurons to choose from by gathering all acceptable
		// neurons from the inputs and processing neuron lists.
		double tempPurity;
		std::vector<JNeuron_Snap*> possibleInputNeurons;
		std::vector<double> chances;

		for (auto nPtr : m_inputNeurons)
		{
			tempPurity = nPtr->getFirePurity(m_correctOutputAction);
			if (tempPurity >= minimumPurity)
			{
				possibleInputNeurons.push_back(nPtr);
				chances.push_back(tempPurity + addedPurity);
			}
		}

		for (auto nPtr : m_processingNeurons)
		{
			tempPurity = nPtr->getFirePurity(m_correctOutputAction);
			if (tempPurity >= minimumPurity)
			{
				possibleInputNeurons.push_back(nPtr);
				chances.push_back(tempPurity + addedPurity);
			}
		}

		// If we don't have enough "pure" inputs, we can't create the neuron:
		if (possibleInputNeurons.size() < m_dendriteMinCountPerNeuron)
		{
			--m_processingNeuronCreatedCount;
			// std::cout << "Failed to create a pure processing neuron due to too few inputs." << std::endl;
			return;
		}

		// Create the neuron:
		unsigned int nNumber = static_cast<unsigned int>(m_allNeurons.size());
		JNeuron_Snap* newNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::PROCESSING, nNumber,
			m_neuronFireThreshold, m_actionSize);

		// There are enough inputs, but not so many that we need to choose among them:
		if (possibleInputNeurons.size() <= m_dendriteStartCountPerNeuron)
		{
			for (auto nPtr : possibleInputNeurons)
			{
				newNeuron->m_inputNeurons.push_back(nPtr->m_neuronNumber);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
				nPtr->m_outputNeurons.push_back(nNumber);
			}
		}
		// More than enough inputs to choose from. Select randomly:
		else
		{
			while (newNeuron->m_inputNeurons.size() < m_dendriteStartCountPerNeuron)
			{
				std::discrete_distribution<std::size_t> dist{ chances.begin(), chances.end() };
				unsigned int neuronNumber;

				// Add until we have the right number of dendrites:
				while (newNeuron->m_inputNeurons.size() < m_dendriteStartCountPerNeuron)
				{
					neuronNumber = possibleInputNeurons[dist(gen)]->m_neuronNumber;

					// Make sure we haven't already selected this one:
					if (newNeuron->m_inputNeurons.end() != std::find(newNeuron->m_inputNeurons.begin(), newNeuron->m_inputNeurons.end(), neuronNumber))
						continue;

					newNeuron->m_inputNeurons.push_back(neuronNumber);
					newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
					m_allNeurons[neuronNumber]->m_outputNeurons.push_back(nNumber);					
				}
			}
		}

		m_processingNeurons.push_back(newNeuron);
		m_allNeurons.push_back(newNeuron);
	}

	void JBrain_Snap::doCreateProcessingNeuron()
	{
		++m_processingNeuronCreatedCount;

		static std::mt19937_64 gen(std::random_device{}());

		// Get a list of all processing and input neurons to choose from:
		std::vector<JNeuron_Snap*> possibleInputNeurons {};
		std::copy(m_inputNeurons.begin(), m_inputNeurons.end(), std::back_inserter(possibleInputNeurons));
		std::copy(m_processingNeurons.begin(), m_processingNeurons.end(), std::back_inserter(possibleInputNeurons));

		// Get how much each of those neurons is used:
		std::vector<unsigned int> inputUsage = getOutputCountVector(possibleInputNeurons);
		unsigned int maxInputUsage = *std::max_element(inputUsage.begin(), inputUsage.end());

		unsigned int nNumber = static_cast<unsigned int>(m_allNeurons.size());
		JNeuron_Snap* newNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::PROCESSING, nNumber,
			m_neuronFireThreshold, m_actionSize);

		// If there aren't enough inputs to satisfy dendrite start count, do our best:
		if (m_dendriteStartCountPerNeuron > static_cast<unsigned int>(possibleInputNeurons.size()))
		{
			for (auto nPtr : possibleInputNeurons)
			{
				newNeuron->m_inputNeurons.push_back(nPtr->m_neuronNumber);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
				nPtr->m_outputNeurons.push_back(nNumber);
			}
		}
		// There are at least enough. Choose randomly.
		else
		{
			while (newNeuron->m_inputNeurons.size() < m_dendriteStartCountPerNeuron)
			{
				std::vector<unsigned int> chances;  // Higher is better:
				for (auto used : inputUsage)
					chances.push_back(maxInputUsage - used + 1);  // Can't have all 0s

				std::discrete_distribution<std::size_t> dist{ chances.begin(), chances.end() };
				unsigned int selection;

				// Add until we have the right number of dendrites:
				while (newNeuron->m_inputNeurons.size() < m_dendriteStartCountPerNeuron)
				{
					selection = possibleInputNeurons[dist(gen)]->m_neuronNumber;
					
					// Make sure we haven't already selected this one:
					if (newNeuron->m_inputNeurons.end() != std::find(newNeuron->m_inputNeurons.begin(), newNeuron->m_inputNeurons.end(), selection))
						continue;

					newNeuron->m_inputNeurons.push_back(selection);
					newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
					m_allNeurons[selection]->m_outputNeurons.push_back(nNumber);
				}
			}
		}

		// HDC neurons need a default fire value:
		if (m_useHDCMode)
			newNeuron->m_fireValue = std::numeric_limits<double>::max();

		m_allNeurons.push_back(newNeuron);
		m_processingNeurons.push_back(newNeuron);
	}

	void JBrain_Snap::createAllPassthroughInputNeurons()
	{
		// Create 1 input neuron for each possible input:
		unsigned int nNumber;
		for (unsigned int i = 0; i < m_observationSize; ++i)
		{
			nNumber = static_cast<unsigned int>(m_allNeurons.size());
			JNeuron_Snap* tempPtr = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::INPUT, nNumber,
				1.0, m_actionSize);
			
			// Record the input we're listening to:
			tempPtr->m_inputNeurons.push_back(i);
			tempPtr->m_inputWeights.push_back(1.0);

			m_allNeurons.push_back(tempPtr);
			m_inputNeurons.push_back(tempPtr);
		}
	}

	void JBrain_Snap::doCreateInputNeuron()
	{
		// Should never be called when using passthrough input neurons:		
		if (m_usePassthroughInputNeurons)
			return;

		++m_inputNeuronCreatedCount;

		// Allocate it once:
		static std::mt19937_64 gen(std::random_device{}());
		
		// Get how much each input has been used so we can favor dendritic connections
		// to less-used inputs:
		std::vector<unsigned int> inputUsage = getUsedInputsCount();
		unsigned int maxInputUsage = *std::max_element(inputUsage.begin(), inputUsage.end());
		
		// Fill inputs with 0..inputSize-1
		std::vector<unsigned int> inputs(m_observationSize);
		std::iota(inputs.begin(), inputs.end(), 0);

		// We always create neurons with neuron # == size of all-neurons-vector:
		JNeuron_Snap* newNeuron = new JNeuron_Snap(CGP::JNEURON_SNAP_TYPE::INPUT,
			static_cast<unsigned int>(m_allNeurons.size()), m_neuronFireThreshold, m_actionSize);

		// If there are only as many possible inputs as we are creating dendrites, one to each:
		if (m_dendriteStartCountPerNeuron >= m_observationSize)
		{
			for (unsigned int i = 0; i < m_observationSize; ++i)
			{
				newNeuron->m_inputNeurons.push_back(i);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
			}
		}
		else // Must be more inputs than dendrites we create, so this loop won't be infinite:
		{
			std::vector<unsigned int> chances;  // Higher is better:
			for (auto used : inputUsage)
				chances.push_back(maxInputUsage - used + 1);  // Can't have all 0s

			std::discrete_distribution<std::size_t> dist{ chances.begin(), chances.end() };
			unsigned int selection;

			// Add until we have the right number of dendrites:
			while (newNeuron->m_inputNeurons.size() < m_dendriteStartCountPerNeuron)
			{
				selection = inputs[dist(gen)];

				// Make sure we haven't already selected this one:
				if (newNeuron->m_inputNeurons.end() != std::find(newNeuron->m_inputNeurons.begin(), newNeuron->m_inputNeurons.end(), selection))
					continue;

				newNeuron->m_inputNeurons.push_back(selection);
				newNeuron->m_inputWeights.push_back(m_dendriteStartingWeight);
			}
		}

		// Add this neuron to the appropriate lists:
		m_allNeurons.push_back(newNeuron);
		m_inputNeurons.push_back(newNeuron);
	}

	void JBrain_Snap::doDestroyInputNeuron(const int& neuronNumber)
	{
		// No changes to input neurons if using passthrough input neurons:
		if (m_usePassthroughInputNeurons)
			return;

		static std::mt19937_64 gen(std::random_device{}());

		// Even if we're told to, we don't destroy the last input neuron:
		if (m_inputNeurons.size() < 2)
			return;
		
		++m_inputNeuronDestroyedCount;

		// No favoring factors implemented yet, pick an input neuron at random if a 
		// neuronNumber wasn't provided:
		if (neuronNumber < 0)
		{
			std::uniform_int_distribution<> dist(0, static_cast<int>(m_inputNeurons.size()) - 1);
			unsigned int ranVal = static_cast<unsigned int>(dist(gen));

			// Let delete handle the details:
			deleteNeuron(m_inputNeurons[ranVal]->m_neuronNumber);
		}
		else
		{
			deleteNeuron(static_cast<const unsigned int>(neuronNumber));
		}

		// We can't ignore an input, if we destroyed an input neuron and no longer are reading
		// all of the inputs, create until we are:
		ensureAllInputsUsed();
	}

	void JBrain_Snap::doDestroyProcessingNeuron(const int& neuronNumber)
	{
		static std::mt19937_64 gen(std::random_device{}());

		// Even if we're told to, we don't destroy the last processing neuron:
		if (m_processingNeurons.size() < 2)
			return;

		++m_processingNeuronDestroyedCount;

		// If a neuron number was provided, destroy it. Otherwise go through the work of selecting
		// a random neuron:
		if (neuronNumber > 0)
		{
			deleteNeuron(static_cast<const unsigned int>(neuronNumber));
			return;
		}

		// Build our vector of favors:
		std::vector<double> chances;
		double startingWeight = 0.01; // Need a non-zero value to start.
		double fullChance, outPercent, inPercent;
		JNeuron_Snap* nPtr;
		for (unsigned int i = 0; i < m_processingNeurons.size(); ++i)
		{
			nPtr = m_processingNeurons[i];
			fullChance = startingWeight;
			if (m_destroyNeuron_FavorFewerConnections)
			{
				// Connections means input AND output. We'll use the maximum dendrite connections IN
				// as the assumed max connections OUT.
				outPercent = fmin(1.0, static_cast<double>(nPtr->m_outputNeurons.size()) / static_cast<double>(m_dendriteMaxCountPerNeuron));
				inPercent = fmin(1.0, static_cast<double>(nPtr->m_inputNeurons.size()) / static_cast<double>(m_dendriteMaxCountPerNeuron));
				
				// Average them together and add them to our weight:
				fullChance += ((outPercent + inPercent) / 2.0);
			}

			// Favory younger neurons means we want MORE points for lower age:
			if (m_destroyNeuron_FavorYoungerNeurons)
			{
				fullChance += 1.0 - (static_cast<double>(nPtr->m_age) / static_cast<double>(m_neuronMaximumAge));
			}

			chances.push_back(fullChance);
		}

		// Now, choose based on this distribution:
		std::discrete_distribution<std::size_t> neuronChoice{ chances.begin(), chances.end() };
		nPtr = m_processingNeurons[neuronChoice(gen)];

		// Let delete handle the details:
		deleteNeuron(nPtr->m_neuronNumber);
	}

	void JBrain_Snap::deleteNeuron(unsigned int neuronNumber)
	{
		JNeuron_Snap* neuronPtr = m_allNeurons[neuronNumber];

		// No changes to input neurons if using passthrough input neurons:		
		if (neuronPtr != nullptr &&
			m_usePassthroughInputNeurons &&
			neuronPtr->m_type == CGP::JNEURON_SNAP_TYPE::INPUT)
		{
			return;
		}

		unsigned int idx;

		// Remove all references:
		for (auto elem : m_allNeurons)
		{
			if (elem == nullptr)
				continue;

			// HDC Proc neurons update their input values elsewhere:
			if (m_useHDCMode && elem->m_type == CGP::JNEURON_SNAP_TYPE::PROCESSING)
				continue;

			// Remove it if it exists as an input:
			auto inIter = std::find(elem->m_inputNeurons.begin(), elem->m_inputNeurons.end(), neuronNumber);
			if (inIter != elem->m_inputNeurons.end())
			{
				idx = static_cast<unsigned int>(inIter - elem->m_inputNeurons.begin());
				elem->m_inputNeurons.erase(elem->m_inputNeurons.begin() + idx);
				elem->m_inputWeights.erase(elem->m_inputWeights.begin() + idx);
			}

			// Remove it if it exists as an output:
			auto outIter = std::find(elem->m_outputNeurons.begin(), elem->m_outputNeurons.end(), neuronNumber);
			if (outIter != elem->m_outputNeurons.end())
			{
				elem->m_outputNeurons.erase(outIter);
			}
		}
		
		// All references from other neurons removed, now make sure it is removed from our neuron lists:		

		// For code simplicity, just search all lists. Need to use vector pointers here
		// to stop it from creating copies and instead modify the real vectors:		
		std::vector<std::vector<JNeuron_Snap*>* > allNeuronLists{&m_inputNeurons, &m_processingNeurons, &m_outputNeurons};
		for (auto elemList : allNeuronLists)
		{
			auto iter = std::find(elemList->begin(), elemList->end(), neuronPtr);				
			if (iter != elemList->end())
				elemList->erase(iter);
		}

		// Finally, free the memory and replace it with a nullptr in the full list:
		delete neuronPtr;
		m_allNeurons[neuronNumber] = nullptr;
	}

	bool JBrain_Snap::setValueByName(const std::string& name, const unsigned int& value)
	{
		bool retVal = true; // Set to false if we don't find the name.

		if (name == "NeuronAccumulationDuration")
			m_neuronAccumulateDuration = value;
		else if (name == "BrainProcessingStepsAllowed")
			m_brainProcessingStepsAllowed = value;
		else if (name == "InitialInputNeuronCount")
			m_initialInputNeuronCount = value;
		else if (name == "InitialProcessingNeuronCount")
			m_initialProcessingNeuronCount = value;
		else if (name == "DendriteMinCountPerNeuron")
			m_dendriteMinCountPerNeuron = value;
		else if (name == "DendriteMaxCountPerNeuron")
			m_dendriteMaxCountPerNeuron = value;
		else if (name == "DendriteStartCountPerNeuron")
			m_dendriteStartCountPerNeuron = value;
		else if (name == "BaseProcessingNeuronCount")
			m_baseProcessingNeuronCount = value;
		else if (name == "ObservationSize")
			m_observationSize = value;
		else if (name == "ActionSize")
			m_actionSize = value;
		else if (name == "HDC_MinimumDeleteDistance")
			m_hdcMinimumDeleteDistance = value;
		else
			retVal = false;

		return retVal;
	}

	bool JBrain_Snap::setValueByName(const std::string& name, const bool& value, const bool& flipBool)
	{
		bool retVal = true; // Set to false if we don't find the name.

		if (name == "NeuronResetOnFiring")
		{
			if (flipBool)
				m_neuronResetOnFiring = !m_neuronResetOnFiring;
			else
				m_neuronResetOnFiring = value;
		}
		else if (name == "NeuronResetAfterOutput")
		{
			if (flipBool)
				m_neuronResetAfterOutput = !m_neuronResetAfterOutput;
			else
				m_neuronResetAfterOutput = value;
		}
		else if (name == "DestroyNeuron_FavorFewerConnections")
		{
			if (flipBool)
				m_destroyNeuron_FavorFewerConnections = !m_destroyNeuron_FavorFewerConnections;
			else
				m_destroyNeuron_FavorFewerConnections = value;
		}
		else if (name == "DestroyNeuron_FavorYoungerNeurons")
		{
			if (flipBool)
				m_destroyNeuron_FavorYoungerNeurons = !m_destroyNeuron_FavorYoungerNeurons;
			else
				m_destroyNeuron_FavorYoungerNeurons = value;
		}
		else if (name == "UsePassthroughInputNeurons")
		{
			if (flipBool)
				m_usePassthroughInputNeurons = !m_usePassthroughInputNeurons;
			else
				m_usePassthroughInputNeurons = value;
		}
		else if (name == "UseHDCMode")
		{
			if (flipBool)
				m_useHDCMode = !m_useHDCMode;
			else
				m_useHDCMode = value;
		}
		else
			retVal = false;

		return retVal;
	}

	bool JBrain_Snap::setValueByName(const std::string& name, const double& value)
	{
		bool retVal = true;

		if (name == "OverallProbability")
			m_overallProbability = value;
		else if (name == "NeuronFireThreshold")
			m_neuronFireThreshold = value;
		else if (name == "DendriteWeightChange")
			m_dendriteWeightChange = value;
		else if (name == "DendriteMinimumWeight")
			m_dendriteMinimumWeight = value;
		else if (name == "DendriteMaximumWeight")
			m_dendriteMaximumWeight = value;
		else if (name == "DendriteStartingWeight")
			m_dendriteStartingWeight = value;
		else if (name == "DendriteWeightTickDownAmount")
			m_dendriteWeightTickDownAmount = value;
		else if (name == "DendriteCorrectWeightChange")
			m_dendriteCorrectWeightChange = value;
		else if (name == "DendriteIncorrectWeightChange")
			m_dendriteIncorrectWeightChange = value;
		else if (name == "StepCreateNeuronChance")
			m_stepCreateNeuronChance = value;
		else if (name == "StepCreateNeuron_BaseCountRatioMultiplier")
			m_stepCreateNeuron_BaseCountRatioMultiplier = value;
		else if (name == "StepCreateInputNeuronChance")
			m_stepCreateInputNeuronChance = value;
		else if (name == "StepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier")
			m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier = value;
		else if (name == "StepDestroyNeuronChance")
			m_stepDestroyNeuronChance = value;
		else if (name == "StepDestroyNeuron_CountBaseRatioMultiplier")
			m_stepDestroyNeuron_CountBaseRatioMultiplier = value;
		else if (name == "StepDestroyInputNeuronChance")
			m_stepDestroyInputNeuronChance = value;
		else if (name == "StepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier")
			m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier = value;
		else if (name == "RunCreateNeuronChance")
			m_runCreateNeuronChance = value;
		else if (name == "RunCreateNeuron_BaseCountRatioMultiplier")
			m_runCreateNeuron_BaseCountRatioMultiplier = value;
		else if (name == "RunCreateInputNeuronChance")
			m_runCreateInputNeuronChance = value;
		else if (name == "RunCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier")
			m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier = value;
		else if (name == "RunDestroyNeuronChance")
			m_runDestroyNeuronChance = value;
		else if (name == "RunDestroyNeuron_CountBaseRatioMultiplier")
			m_runDestroyNeuron_CountBaseRatioMultiplier = value;
		else if (name == "RunDestroyInputNeuronChance")
			m_runDestroyInputNeuronChance = value;
		else if (name == "RunDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier")
			m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier = value;
		else if (name == "OutputPositive_CascadeProbability")
			m_outputPositive_CascadeProbability = value;
		else if (name == "OutputPositive_InSequence_IncreaseDendriteWeight")
			m_outputPositive_InSequence_IncreaseDendriteWeight = value;
		else if (name == "OutputPositive_NoConnection_InSequence_CreateConnection")
			m_outputPositive_NoConnection_InSequence_CreateConnection = value;
		else if (name == "OutputPositive_YesFire_UnusedInput_DecreaseWeight")
			m_outputPositive_YesFire_UnusedInput_DecreaseWeight = value;
		else if (name == "OutputPositive_YesFire_UnusedInput_BreakConnection")
			m_outputPositive_YesFire_UnusedInput_BreakConnection = value;
		else if (name == "OutputNegative_CascadeProbability")
			m_outputNegative_CascadeProbability = value;
		else if (name == "OutputNegative_InSequence_DecreaseDendriteWeight")
			m_outputNegative_InSequence_DecreaseDendriteWeight = value;
		else if (name == "OutputNegative_InSequence_BreakConnection")
			m_outputNegative_InSequence_BreakConnection = value;
		else if (name == "OutputNegative_CreatePureProcessingNeuron")
			m_outputNegative_CreatePureProcessingNeuron = value;
		else if (name == "NoOutput_IncreaseInputDendriteWeight")
			m_noOutput_IncreaseInputDendriteWeight = value;
		else if (name == "NoOutput_AddProcessingNeuronDendrite")
			m_noOutput_AddProcessingNeuronDendrite = value;
		else if (name == "NoOutput_IncreaseProcessingNeuronDendriteWeight")
			m_noOutput_IncreaseProcessingNeuronDendriteWeight = value;
		else if (name == "NoOutput_AddOutputNeuronDendrite")
			m_noOutput_AddOutputNeuronDendrite = value;
		else if (name == "NoOutput_IncreaseOutputNeuronDendriteWeight")
			m_noOutput_IncreaseOutputNeuronDendriteWeight = value;
		else if (name == "NoOutput_CreateProcessingNeuron")
			m_noOutput_CreateProcessingNeuron = value;
		else if (name == "NoOutput_CreatePureProcessingNeuron")
			m_noOutput_CreatePureProcessingNeuron = value;
		else
			retVal = false;

		return retVal;
	}

	void JBrain_Snap::writeSelfToJson(json& j)
	{
		j["name"] = m_name;
		j["parentName"] = m_parentName;

		j["staticOverallProbability"] = m_staticOverallProbability;
		j["overallProbability"] = m_overallProbability;
		j["dynamicProbabilityUsage"] = CGP::DynamicProbabilityToString(m_dynamicProbabilityUsage);
		j["dynamicProbabilityMultiplier"] = m_dynamicProbabilityMultiplier;
		j["mostRecentScorePercent"] = m_mostRecentScorePercent;
		j["neuronAccumulateDuration"] = m_neuronAccumulateDuration;
		j["neuronResetOnFiring"] = m_neuronResetOnFiring;
		j["neuronResetAfterOutput"] = m_neuronResetAfterOutput;
		j["neuronFireThreshold"] = m_neuronFireThreshold;
		j["neuronMaximumAge"] = m_neuronMaximumAge;
		j["brainProcessingStepsAllowed"] = m_brainProcessingStepsAllowed;
		j["usePassthroughInputNeurons"] = m_usePassthroughInputNeurons;

		// Stored so we can pass them in a copy constructor:
		j["initialInputNeuronCount"] = m_initialInputNeuronCount;
		j["initialProcessingNeuronCount"] = m_initialProcessingNeuronCount;
		j["maximumProcessingNeuronCount"] = m_maximumProcessingNeuronCount;
		j["maximumInputNeuronsToInputRatio"] = m_maximumInputNeuronsToInputRatio;

		// Dendrite-specific variables:
		j["dendriteWeightChange"] = m_dendriteWeightChange;
		j["dendriteMinimumWeight"] = m_dendriteMinimumWeight;
		j["dendriteMaximumWeight"] = m_dendriteMaximumWeight;
		j["dendriteStartingWeight"] = m_dendriteStartingWeight;
		j["dendriteWeightTickDownAmount"] = m_dendriteWeightTickDownAmount;
		j["dendriteCorrectWeightChange"] = m_dendriteCorrectWeightChange;
		j["dendriteIncorrectWeightChange"] = m_dendriteIncorrectWeightChange;
		j["dendriteMinCountPerNeuron"] = m_dendriteMinCountPerNeuron;
		j["dendriteMaxCountPerNeuron"] = m_dendriteMinCountPerNeuron;
		j["dendriteStartCountPerNeuron"] = m_dendriteStartCountPerNeuron;

		// Base neuron count is used to regulate the neuron-creation and neuron-destruction chances:
		j["baseProcessingNeuronCount"] = m_baseProcessingNeuronCount;

		// Environment-related values:
		j["observationSize"] = m_observationSize;
		j["actionSize"] = m_actionSize;

		// Step-Events:
		j["stepCreateNeuronChance"] = m_stepCreateNeuronChance;
		j["stepCreateNeuron_BaseCountRatioMultiplier"] = m_stepCreateNeuron_BaseCountRatioMultiplier;
		j["stepCreateInputNeuronChance"] = m_stepCreateInputNeuronChance;
		j["stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier"] = m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier;
		j["stepDestroyNeuronChance"] = m_stepDestroyNeuronChance;
		j["stepDestroyNeuron_CountBaseRatioMultiplier"] = m_stepDestroyNeuron_CountBaseRatioMultiplier;
		j["stepDestroyInputNeuronChance"] = m_stepDestroyInputNeuronChance;
		j["stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier"] = m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier;
		
		// Weighting neuron destruction:
		j["DestroyNeuron_FavorFewerConnections"] = m_destroyNeuron_FavorFewerConnections;
		j["DestroyNeuron_FavorYoungerNeurons"] = m_destroyNeuron_FavorYoungerNeurons;

		// Run-Events:
		j["runCreateNeuronChance"] = m_runCreateNeuronChance;
		j["runCreateNeuron_BaseCountRatioMultiplier"] = m_runCreateNeuron_BaseCountRatioMultiplier;
		j["runCreateInputNeuronChance"] = m_runCreateInputNeuronChance;
		j["runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier"] = m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier;
		j["runDestroyNeuronChance"] = m_runDestroyNeuronChance;
		j["runDestroyNeuron_CountBaseRatioMultiplier"] = m_runDestroyNeuron_CountBaseRatioMultiplier;
		j["runDestroyInputNeuronChance"] = m_runDestroyInputNeuronChance;
		j["runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier"] = m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier;

		// Output-Positive Events:
		j["outputPositive_CascadeProbability"] = m_outputPositive_CascadeProbability;
		j["outputPositive_InSequence_IncreaseDendriteWeight"] = m_outputPositive_InSequence_IncreaseDendriteWeight;
		j["outputPositive_NoConnection_InSequence_CreateConnection"] = m_outputPositive_NoConnection_InSequence_CreateConnection;
		j["outputPositive_YesFire_UnusedInput_DecreaseWeight"] = m_outputPositive_YesFire_UnusedInput_DecreaseWeight;
		j["outputPositive_YesFire_UnusedInput_BreakConnection"] = m_outputPositive_YesFire_UnusedInput_DecreaseWeight;

		// Output-Negative Events:
		j["outputNegative_CascadeProbability"] = m_outputNegative_CascadeProbability;
		j["outputNegative_InSequence_DecreaseDendriteWeight"] = m_outputNegative_InSequence_DecreaseDendriteWeight;
		j["outputNegative_InSequence_BreakConnection"] = m_outputNegative_InSequence_BreakConnection;
		j["outputNegative_CreatePureProcessingNeuron"] = m_outputNegative_CreatePureProcessingNeuron;

		// No Output events:
		j["noOutput_IncreaseInputDendriteWeight"] = m_noOutput_IncreaseInputDendriteWeight;
		j["noOutput_AddProcessingNeuronDendrite"] = m_noOutput_AddProcessingNeuronDendrite;
		j["noOutput_IncreaseProcessingNeuronDendriteWeight"] = m_noOutput_IncreaseProcessingNeuronDendriteWeight;
		j["noOutput_AddOutputNeuronDendrite"] = m_noOutput_AddOutputNeuronDendrite;
		j["noOutput_IncreaseOutputNeuronDendriteWeight"] = m_noOutput_IncreaseOutputNeuronDendriteWeight;
		j["noOutput_CreateProcessingNeuron"] = m_noOutput_CreateProcessingNeuron;
		j["noOutput_CreatePureProcessingNeuron"] = m_noOutput_CreatePureProcessingNeuron;

		// HDC-specific variables:
		j["UseHDCMode"] = m_useHDCMode;
		j["HDC_MinimumDeleteDistance"] = m_hdcMinimumDeleteDistance;
		j["HDC_LearnMode"] = CGP::HDCLearnModeToString(m_hdcLearnMode);
		j["HDC_LearnMode_Original"] = CGP::HDCLearnModeToString(m_hdcLearnMode_original);

		j["allNeurons_size"] = m_allNeurons.size();
		j["inputNeurons_size"] = m_inputNeurons.size();
		j["processingNeurons_size"] = m_processingNeurons.size();
		j["outputNeurons_size"] = m_outputNeurons.size();				
	}

	bool JBrain_Snap::setValueByName(const std::string& name, std::string value)
	{
		bool retVal = true;
		if (name == "DynamicProbabilityUsage")
			m_dynamicProbabilityUsage = CGP::StringToDynamicProbability(value);
		else if (name == "HDCLearnMode")
			m_hdcLearnMode = CGP::StringToHDCLearnMode(value);
		else if (name == "Name")
			m_name = value;
		else
			retVal = false;

		return retVal;
	}

	JBrain_Snap::JBrain_Snap(const JBrain_Snap& other)
		: JBrain_Snap(
			other.m_name,
			other.m_name,
			other.m_overallProbability,
			other.m_dynamicProbabilityUsage,
			other.m_dynamicProbabilityMultiplier,
			other.m_neuronAccumulateDuration,
			other.m_neuronResetOnFiring,
			other.m_neuronResetAfterOutput,
			other.m_neuronFireThreshold,
			other.m_neuronMaximumAge,
			other.m_brainProcessingStepsAllowed,
			other.m_dendriteWeightChange,
			other.m_dendriteMinimumWeight,
			other.m_dendriteMaximumWeight,
			other.m_dendriteStartingWeight,
			other.m_dendriteWeightTickDownAmount,
			other.m_dendriteCorrectWeightChange,
			other.m_dendriteIncorrectWeightChange,
			other.m_dendriteMinCountPerNeuron,
			other.m_dendriteMaxCountPerNeuron,
			other.m_dendriteStartCountPerNeuron,
			other.m_baseProcessingNeuronCount,
			other.m_actionSize,
			other.m_initialInputNeuronCount,
			other.m_initialProcessingNeuronCount,
			other.m_maximumProcessingNeuronCount,
			other.m_maximumInputNeuronsToInputRatio,
			other.m_stepCreateNeuronChance,
			other.m_stepCreateNeuron_BaseCountRatioMultiplier,
			other.m_stepCreateInputNeuronChance,
			other.m_stepCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			other.m_stepDestroyNeuronChance,
			other.m_stepDestroyNeuron_CountBaseRatioMultiplier,
			other.m_stepDestroyInputNeuronChance,
			other.m_stepDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier,
			other.m_destroyNeuron_FavorFewerConnections,
			other.m_destroyNeuron_FavorYoungerNeurons,
			other.m_runCreateNeuronChance,
			other.m_runCreateNeuron_BaseCountRatioMultiplier,
			other.m_runCreateInputNeuronChance,
			other.m_runCreateInputNeuron_ObservationSizeInputNeuronRatioMultiplier,
			other.m_runDestroyNeuronChance,
			other.m_runDestroyNeuron_CountBaseRatioMultiplier,
			other.m_runDestroyInputNeuronChance,
			other.m_runDestroyInputNeuron_InputNeuronObservationSizeRatioMultiplier,
			other.m_outputPositive_CascadeProbability,
			other.m_outputPositive_InSequence_IncreaseDendriteWeight,
			other.m_outputPositive_NoConnection_InSequence_CreateConnection,
			other.m_outputPositive_YesFire_UnusedInput_DecreaseWeight,
			other.m_outputPositive_YesFire_UnusedInput_BreakConnection,
			other.m_outputNegative_CascadeProbability,
			other.m_outputNegative_InSequence_DecreaseDendriteWeight,
			other.m_outputNegative_InSequence_BreakConnection,
			other.m_outputNegative_CreatePureProcessingNeuron,
			other.m_noOutput_IncreaseInputDendriteWeight,
			other.m_noOutput_AddProcessingNeuronDendrite,
			other.m_noOutput_IncreaseProcessingNeuronDendriteWeight,
			other.m_noOutput_AddOutputNeuronDendrite,
			other.m_noOutput_IncreaseOutputNeuronDendriteWeight,
			other.m_noOutput_CreateProcessingNeuron,
			other.m_noOutput_CreatePureProcessingNeuron,
			other.m_usePassthroughInputNeurons,
			other.m_useHDCMode,
			other.m_hdcMinimumDeleteDistance,
			other.m_hdcLearnMode_original,
			other.m_observationProcessor)
	{}

} // End JBrain Namespace