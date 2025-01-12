#pragma once
#include "Enums.h"
#include <vector>
namespace JBrain {

  // Observation processor takes in the environment observation and produces the observation that the
  // brain will use when processing. It also processes outputs, but its name remains just
  // ObservationProcessor to avoid an error-prone code update.
  class ObservationProcessor
  {
  public:
    ObservationProcessor(CGP::INPUT_PREPROCESSING procType, const unsigned int& obsSize,      
      const std::vector<std::vector<double> >& obsRanges = {},
      const std::vector<unsigned int>& obsBucketsPerInput = {},
      const unsigned int& actionSize = 0,
      const std::vector<std::vector<double> >& actionRanges = {},
      const std::vector<unsigned int>& actionBucketsPerInput = {});

    std::vector<double> processInput(const std::vector<double>& envObs);
    // Process outputs in bucket style. Get a vector of which bucket was selected and convert
    // it into the appropriate floating point value:
    std::vector<double> processOutput(const std::vector<unsigned int>& brainAct);

    unsigned int getExpectedOutputSize() { return m_obsOutputSize; }    
    std::vector<std::vector<double> > getSeparatedInputs() { return m_separatedInputs; }
    std::vector<unsigned int> getSeparatedInputs_simplified() { return m_separatedInputs_simplified; }
    std::vector<unsigned int> getSeparatedOutputs_simplified(const std::vector<double>& outputs);
    std::vector<unsigned int> getActionBucketSizes() { return m_individualActSizes; }

  private:
    // processInput produces the full, combined bucket input. For HDC mode, we need
    // to also access each input's separated portion.
    std::vector<std::vector<double> > m_separatedInputs;

    // HDC generally deals with single positive outputs. Make it easier to access those:
    std::vector<unsigned int> m_separatedInputs_simplified;
    
    // Create the bucket-maximums for a single input.
    std::vector<double> createSingleInputBucketCutoff(const double& minVal, const double& maxVal, const unsigned int& buckets);
    
    // Process inputs in different ways:
    std::vector<double> processInput_NoChange(const std::vector<double>& envObs);
    std::vector<double> processInput_Bucket(const std::vector<double>& envObs);
    std::vector<double> processInput_NegativeValueAdd(const std::vector<double>& envObs);

    // Process outputs in different ways:
    std::vector<double> processOutput_Bucket(const std::vector<unsigned int>& brainAct);

    CGP::INPUT_PREPROCESSING m_procType;
    
    // Observation processing:
    unsigned int m_obsSize;
    std::vector<unsigned int> m_individualObsSizes;
    unsigned int m_obsOutputSize;
    std::vector<std::vector<double> > m_obsRanges;
    std::vector<std::vector<double> > m_obsBucketCutoffs;

    // Action processing:
    unsigned int m_actSize;
    std::vector<unsigned int> m_individualActSizes;
    unsigned int m_actOutputSize;
    std::vector<std::vector<double> > m_actRanges;
    std::vector<std::vector<double> > m_actBucketCutoffs;
  };

} // End namespace JBrain

