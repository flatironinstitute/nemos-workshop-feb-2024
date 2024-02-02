# Feb 5, 2024 Workshop 

We will be hosting a workshop about the [Nemos package](https://github.com/flatironinstitute/nemos) on Monday Feb 5, 2024. Nemos is a statistical modeling framework for systems neuroscience, leveraging [jax](https://jax.readthedocs.io/) to provide GPU-accelerated implementations of standard analyses. Our first implemented model is the Generalized Linear Model (GLM) for analyzing spiking data and, in addition to providing well-tested implementations of the models, we intend to provide pedagogical materials to explain how to understand and interpret their results.

In this first workshop, we will provide an introduction to the basic functionality of the package, using it to analyze data from a variety of domains within neuroscience. We are not ready to release a version 1.0 of the package (i.e., we are not promising the interface won't change); our primary goal at this point is to receive user feedback: are we explaining the concepts well? does interacting with the objects feel reasonable and "scikit-learn-like", or does it feel stilted? are our docstrings clear? etc.

While we will eventually want Nemos to be widely-used in the systems neuroscience community, by experimentalists and theorists with a wide variety of programming knowledge, for this first workshop, we would like attendees who have worked with spiking data before, are somewhat familiar with python, and have used computational models of some type.

The workshop will run from 9am to 5pm at the Flatiron Institute Center for Computational Neuroscience, with breakfast starting at 8am, lunch catered at noon, and dinner offsite at 6pm.

## Pre-workshop setup

For this workshop, please bring your own laptop. We will be working in jupyter notebooks on a Flatiron-hosted [binder](https://mybinder.readthedocs.io/en/latest/index.html) instance. This means that you do not need to install or download anything to your personal machine; everything will be hosted on the Flatiron cluster and everyone should have access to a GPU.

In the week before the workshop, please visit [the link to the binder instance](https://binder.flatironinstitute.org/~wbroderick/nemos?filepath=notebooks) to see if you have access. You will have to log in with your google account (the one you registered with). If you get a 403 Forbidden error, make sure you selected the right google account and, if so, let Billy know.
