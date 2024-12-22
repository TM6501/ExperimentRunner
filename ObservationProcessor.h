#pragma once
#include "Enums.h"
#include <vector>
namespace JBrain {

  // Observation processor takes in the environment observation and produces the observation that the
  // brain will use when processing:
  class ObservationProcessor
  {
  public:
    ObservationProcessor(CGP::INPUT_PREPROCESSING procType, const unsigned int& obsSize,
      const std::vector<std::vector<double> >& ranges = {}, const unsigned int& bucketsPerInput = 0);

    std::vector<double> processInput(const std::vector<double>& envObs);

    unsigned int getExpectedOutputSize() { return m_obsOutputSize; }
  private:
    // Create the bucket-maximums for a single input.
    std::vector<double> createSingleInputBucketCutoff(const double& minVal, const double& maxVal, const unsigned int& buckets);
    
    // Process inputs in different ways:
    std::vector<double> processInput_NoChange(const std::vector<double>& envObs);
    std::vector<double> processInput_Bucket(const std::vector<double>& envObs);
    std::vector<double> processInput_NegativeValueAdd(const std::vector<double>& envObs);

    CGP::INPUT_PREPROCESSING m_procType;
    unsigned int m_obsSize;
    unsigned int m_obsOutputSize;
    std::vector<std::vector<double> > m_ranges;
    std::vector<std::vector<double> > m_bucketCutoffs;
  };

} // End namespace JBrain

