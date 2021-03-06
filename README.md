# Neuron Decoding Project 🧠

# Coding To-Do 💻
1. Need to find a better model than Resnet18
2. Keep trying different models and parameters to maximize 6-region classification accuracy -45%, 100epochs, around 60% should be good (classifying 6 regions)

# Writing To-Do 📓
https://www.brainmeeting2021.com/en/event-submissions
1. Add figures
2. Update it based on progress
3. Methods section
4. Read articles found to get a better idea on how to write intro

# Bugs 🦋
1.  Run captum analysis on 45% - can't run one analysis but rest are fine
2.  Implement neuron attribution from CAPTUM -have to build the pipeline, no tutorial with a smooth transition

# Helpful Links 🆘
1. https://pytorch-lightning.readthedocs.io/en/latest/extensions/metrics.html#classification-metrics
2. https://forums.fast.ai/t/determining-when-you-are-overfitting-underfitting-or-just-right/7732/2
3. https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/conv_sequential_example.py
4. https://github.com/pytorchbearer/torchbearer

# Non-urgent 💤
1. Work on writing a program to do descriptive statistics (e.g. range, mean, stdev) that can describe potential differences between the different classifications of images (visp, visal, visam, visrl, vispm, visl).
2. Work on another shuffling method of data --bootstrapping in which we can also get the mean and stdev of its results once we have 10 results from the 10 different trials that were ran.
