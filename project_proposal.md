Project Proposal: Extending the
Forward-Forward Algorithm
-
-
-
Nikhilesh Myanapuri, 22110162
Birudugadda Srivibhav, 22110050
Srivathsa Vamsi Chaturvedula, 22110260
-
Eswar Ganesh Koleti, 22110123
1. Project Overview & Motivation
Geoﬀrey Hinton’s Forward-Forward (FF) algorithm introduces a biologically plausible
alternative to backpropagation by utilizing two forward passes, one with positive (real) data
and one with negative (generated) data. While the initial results on MNIST and CIFAR-10 are
promising, the foundational paper leaves several architectural and mathematical design choices
largely unexplored.
This project aims to systematically investigate three specific open questions regarding the FF
algorithm
's objective functions, activation layers, and spatial processing capabilities. If time
permits, we will also explore the concept of "sleep" (oﬄine negative data processing) to further
model biological learning.
2. Project Objectives
1. Optimize the Goodness Function: Transition from maximizing the sum of squared
activities for positive data to alternative formulations, specifically evaluating the
minimization of unsquared activities on positive data (and maximizing on negative
data).
2. Explore Novel Activation Functions: Replace standard Rectified Linear Units (ReLUs)
with alternative non-linearities. The primary focus will be implementing an activation
function based on the negative log of the density under a t-distribution.
3. Implement Local Goodness Functions for Spatial Data: Instead of a single global
goodness score per layer, we will divide layers into spatial blocks (inspired by Löwe et
al., 2019) that independently calculate goodness. This aims to drastically accelerate the
learning rate for image data.
4. Investigate "Sleep" Phases (Stretch Goal): Attempt to decouple the positive and
negative passes temporally, training on positive data while "awake" and using the
network's own generations to train on negative data while "asleep"
.
3. Implementation Plan
Week 1: Baseline Setup & Goodness Function Variations
●
Tasks:
○
○
○
Replicate a baseline FF multi-layer network (in PyTorch or JAX) and achieve the
paper
's ~1.36% error rate on the MNIST dataset.
Implement alternative goodness functions.
Crucial Implementation Detail: When using unsquared activities, modify the layer
normalization to normalize the sum of the activities rather than the sum of their
squares (as noted in the paper
's footnotes).
Week 2: Activation Function Engineering
●
Tasks:
○
○
○
Implement custom activation layers.
Formulate the mathematical forward pass and local derivatives for the negative
log of the density under a t-distribution.
Train the network using these new activations and compare stability and
convergence speed against the ReLU baseline.
Week 3: Localized Goodness for Spatial Data
●
Tasks:
○
○
○
Adapt the network architecture for spatially structured data (e.g., using localized
receptive fields without weight sharing).
Divide the hidden layers into sub-grids/blocks. Force each block to individually
use the length of its pre-normalized activity vector to distinguish between
positive and negative data.
Test this on CIFAR-10 and jittered MNIST to see if local constraints inject
information into the weights faster than a global constraint.
Week 4: Evaluation, Ablation, & "Sleep" (Stretch Goal)
●
Tasks:
○
○
Consolidate results into a final comparative analysis (ablation study of what
combinations of goodness/activation/spatial-locality work best).
Stretch Goal: Modify the training loop to alternate between thousands of positive
updates (wake) and negative updates (sleep). Experiment with low learning rates
and high momentum to prevent catastrophic forgetting or network collapse
during the "sleep" phase.
4. Evaluation Methodology
●
●
Datasets: We will use MNIST (standard and spatial-permutation invariant) and
CIFAR-10 to allow for direct comparisons against the benchmarks established in
Hinton
's paper.
Evaluation Metrics:
1. Test Error Rate (%): To measure if the generalizations improve or degrade.
2. Convergence Speed: Measured in the number of epochs (and raw wall-clock
time) required to reach a specific error threshold. We expect the local goodness
functions (Objective 3) to significantly reduce required epochs.
3. Representation Quality: Using single-pass softmax evaluation vs. multi-iteration
accumulated goodness evaluation to see how robust the learned representations
are.
5. Expected Outcomes
By the end of the month, this project will yield a comprehensive empirical study of the
Forward-Forward algorithm
's flexibility. We will determine whether replacing ReLUs with
statistical density functions, altering the goodness metrics, and enforcing spatial locality can
make FF a more competitive alternative to standard backpropagation in computer vision tasks.