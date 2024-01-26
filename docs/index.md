# Feb 5, 2024 Workshop 

We will be hosting a workshop about the [Nemos package](https://github.com/flatironinstitute/nemos) on Monday Feb 5, 2024. Nemos is a statistical modeling framework for systems neuroscience, leveraging [jax](https://jax.readthedocs.io/) to provide GPU-accelerated implementations of standard analyses. Our first implemented model is the Generalized Linear Model (GLM) for analyzing spiking data and, in addition to providing well-tested implementations of the models, we intend to provide pedagogical materials to explain how to understand and interpret their results.

In this first workshop, we will provide an introduction to the basic functionality of the package, using it to analyze data from a variety of domains within neuroscience. We are not ready to release a version 1.0 of the package (i.e., we are not promising the interface won't change); our primary goal at this point is to receive user feedback: are we explaining the concepts well? does interacting with the objects feel reasonable and "scikit-learn-like", or does it feel stilted? are our docstrings clear? etc.

While we will eventually want Nemos to be widely-used in the systems neuroscience community, by experimentalists and theorists with a wide variety of programming knowledge, for this first workshop, we would like attendees who have worked with spiking data before, are somewhat familiar with python, and have used computational models of some type.

The workshop will run from 9am to 5pm at the Flatiron Institute Center for Computational Neuroscience, with breakfast starting at 8am, lunch catered at noon, and dinner offsite at 6pm.

## Pre-workshop setup

For this workshop, please bring your own laptop. We will be working in jupyter notebooks on a Flatiron-hosted [binder](https://mybinder.readthedocs.io/en/latest/index.html) instance. This means that you do not need to install or download anything to your personal machine; everything will be hosted on the Flatiron cluster and everyone should have access to a GPU.

In the week before the workshop, please visit [the link to the binder instance](https://binder.flatironinstitute.org/~wbroderick/nemos) to see if you have access. You will have to log in with your google account (the one you registered with). If you get a 403 Forbidden error, make sure you selected the right google account and, if so, let Billy know.

## Introductory Remarks
First of all, thank you to everyone for being here and for being the first alpha testers of Nemos. The aim of this workshop is to collect feedback on both the package and the tutorials, which will serve as teaching material and be integrated into our documentation.

Was it easy to interact with Nemos objects? Is the documentation clear and understandable? Did it provide all the necessary information to set up your model and debug the code? Were the tutorials helpful? What can be improved? We welcome any feedback on your experience.

This workshop is intended for theoretical and experimental neuroscientists who are familiar with Python and NumPy but may lack background knowledge in Generalized Linear Models (GLMs). It is organized into different sections, each lasting approximately one hour and centered around a specific tutorial in the form of an interactive notebook. Theory and exercises will be interleaved, aiming to be gradual and not overwhelming.

We will start with a "Tutorial 0", which should bring everyone up to speed with the GLM framework and with time series manipulations through the "Pynapple" package. Most of you will already be familiar with it, so if we manage to confuse you, that's valuable feedback and a concerning sign!

Next, we will introduce various aspects of the package while analyzing real datasets from various neuroscience domains. The tutorials should feel incremental in terms of complexity and should cover most standard use cases. 

By the end of the exercises, users should feel confident in setting up a GLM with Nemos on their own data and assessing fit quality.

We will reserve some time between 4:30pm and 5pm to collect and discuss your feedback, and then we can call it a day and head out to the restaurant!

Again, thank you all for participating.

The CNS Data Scientists

## Schedule

| Topic                                     | Time           | Instructor Name    |
|-------------------------------------------|----------------|--------------------|
| BREAKFAST                                 | 8 - 9 AM       | FoodTrends Drop Off|
| Welcome                                   | 9 - 9:10 AM    | Edoardo Balzani    |
| The Math (Convolution/Evaluation, Log-likelihood) | 9:10 - 10 AM | Edoardo Balzani    |
| BREAK                                     | 10 - 10:30 AM  |                    |
| Crash course on Pynapple and NWB          | 10:30 - 11 AM  | Guillaume Viejo    |
| Exercise 1 : Current Injection            | 11 AM - 12 PM  | Billy Broderick                |
| LUNCH                                     | 12 - 1 PM      | Pippali Drop Off   |
| Exercise 2 : Head-direction cells         | 1 - 2 PM       | Edoardo Balzani                |
| Exercise 3 : Grid cells          | 2 - 3 PM       | TAs                 |
| BREAK                                     | 3 - 3:30 PM    |                    |
| Exercise 4 :                              | 3:30 - 4:30 PM | TAs                |
| Conclusion/Feedback                       | 4:30 - 5 PM    | TAs                |
| DINNER                                    | 6 PM           |                    |

