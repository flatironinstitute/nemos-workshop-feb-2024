# nemos-workshop-feb-2024

Materials for nemos workshop, Feb 2024

[OSF project](https://osf.io/5crqj/) for uploading NWB files.

## Setting up binder

For this workshop, we're using the [Flatiron
Binder](https://wiki.flatironinstitute.org/SCC/BinderHub). To set that up,
someone with Flatiron cluster access must:

1. Create a directory at `~/public_binder/nemos` (the last directory will be the
   name of the environment).
2. Create a `.public_binder` file with the following contents (note that this is a yaml file):
   ```
   gpu: true
   mem_guarantee: 20G
   mem_limit: 41G
   cpu_guarantee: 2
   cpu_limit: 5
   ```
   
   The mem/cpu guarantee and limit determine how much of each resource every
   user will have available. Each user will have *at least* the guarantee and
   *at most* the limit. In practice, your total number of users multiplied by
   the **limit** should all fit on a single node. Talk with SCC to get a sense
   for the numbers on this, you may need to tweak the above numbers, which
   should work for 18 users on a single GPU node (for this workshop, we have one
   with 750GB memory and 96 cores). See the wiki page linked above for details,
   you may need to talk some with SCC.
   
3. Add `users:` with a list of email addresses for your users. Each user will
   have to login with this google account to gain access. NOTE: this has to be
   the email address that people *use to login*, i.e., aliases are not allowed.
   In particular, this means that NYU email addresses should be their netID form
   (e.g., fl123@nyu.edu), not the expanded one (e.g.,
   firstnamelastname@nyu.edu).
   
4. Clone this workshop repo into your `~/public_binder/nemos` directory and
   symlink the contents of the `binder/` directory:
   
   ```bash
   cd ~/public_binder/nemos
   git clone git@github.com:flatironinstitute/nemos-workshop-feb-2024.git
   # note this must be relative paths! not absolute!
   ln -sv nemos-workshop-feb-2024/binder/* .
   ```
   
   One note here! For our [FENS 2024
   workshop](https://github.com/flatironinstitute/ccn-software-fens-2024), I
   realized that this symlink doesn't work for the `start` script, which may be
   useful for, e.g., setting environmental variables that tell `pooch` where to
   download data. In that case, you must copy the actual start file: `cp
   ccn-software-fens-2024/binder/start .` See the FENS workshop repo for an
   example.

5. In your browser, go to `https://binder.flatironinstitute.org` and enter the
   username of whoever create the directory as the owner, `nemos` as the
   project, and `notebooks/` as the path. This is equivalent to going to
   `https://binder.flatironinstitute.org/~USER/nemos` (I can't figure out how to
   get the path in the url).

NOTE: the inclusion of both `environment.yml` and `environment-cuda.yml` is
because of an issue getting jax working with cuda in binder. With both files,
`environment.yml` gets set up first, which installs conda, python, pip, and
cuda, and then the first line in our `postBuild` will install
`environment-cuda.yml` which installs all our dependencies (including jax).

If you need to force the container to rebuild, because you modified the
requirements or the contents of `binder/`, touch the directory on the cluster:
`touch ~/public_binder/nemos` (see
[wiki](https://wiki.flatironinstitute.org/SCC/BinderHub#Updating_environments)).

In order to install the requirements listed in this repository, we had to
structure it as a installable library. That is, we needed to include a
`pyproject.toml` and put our code within `src/` and everything, so that pip
knows how to install it when passed
`git+https://github.com/flatironinstitute/nemos-workshop-feb-2024.git`
