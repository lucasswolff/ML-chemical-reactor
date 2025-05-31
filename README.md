# PFR Chemical Reactor Neural Network Optimization

## Overview
This project explores the intersection of Machine Learning and Chemical Engineering by developing an AI-driven optimization approach for chemical reactor processes. 
The focus is on using Neural Networks to model and optimize a Plug Flow Reactor (PFR) system, demonstrating how modern ML techniques can complement and potentially enhance traditional chemical engineering methods.

## Motivation
Chemical process optimization traditionally relies on complex engineering equations and simulations that often:
- Require significant computational resources
- Need mathematical approximations to be solvable
- May not fully represent real-world conditions
- Take considerable time to solve

This project proposes an alternative approach using Machine Learning, specifically Neural Networks, to:
- Model complex chemical reactions more efficiently
- Find optimal operating conditions quickly
- Handle non-linear relationships effectively
- Provide accurate predictions while respecting operational constraints

## Methodology

### 1. Data Generation
- Synthetic data generation using fundamental chemical engineering equations
- Creation of thousands of different operational scenarios
- Variables included:
  - Input parameters (temperature, pressure, raw material flow)
  - Output parameters (reaction yield, product concentration)
  - Process constraints and boundaries

### 2. Neural Network Development
- Implementation of a Neural Network model to:
  - Map relationships between input and output variables
  - Learn complex patterns in the reaction system
  - Predict reaction yields under different conditions

### 3. Optimization Framework
- Integration of optimization algorithms with the trained Neural Network
- Implementation of constraints to ensure:
  - Safe operating conditions
  - Feasible solutions within equipment limitations
  - Maximum efficiency while maintaining process stability

## Results

### Model Performance
- Successfully replicated traditional engineering equation results
- Demonstrated high accuracy in predicting reaction outputs
- Validated the feasibility of using ML for process simulation

### Optimization Achievements
- Rapid convergence to optimal operating conditions
- Successful identification of maximum reaction yield points
- Maintained all process constraints and boundaries
- Significantly reduced computation time compared to traditional methods

## Key Advantages
1. **Speed**: Optimization results achieved in seconds
2. **Accuracy**: Comparable results to traditional engineering methods
3. **Flexibility**: Easy adaptation to different process conditions
4. **Practical Application**: Ready for real-world implementation

## Future Work
- Integration with real-time process data
- Extension to other types of chemical reactors
- Implementation of more complex reaction systems
- Development of a user-friendly interface for process engineers

## Technologies Used
- Python
- Machine Learning Libraries
- Optimization Algorithms
- Chemical Engineering Principles

## Author
Lucas Wolff
