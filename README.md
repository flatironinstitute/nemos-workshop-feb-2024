# nemos-workshop-feb-2024

Materials for nemos workshop, Feb 2024

[OSF project](https://osf.io/5crqj/) for uploading NWB files.

## Setting up binder

For this workshop, we're using the [Flatiron
Binder](https://wiki.flatironinstitute.org/SCC/BinderHub). To set that up,
someone with Flatiron cluster access must:

1. Create a directory at `~/public_binder/nemos` (the last directory will be the
   name of the environment).
2. Create a `data/` folder within that directory.
2. Create a `.public_binder` file with the following contents (this is a yaml file):
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
   should work for 18 users on a single GPU node (with 750GB memory and 96
   cores).
   
3. Add `users:` with a list of email addresses for your users. Each user will
   have to login with this google account to gain access.
   
4. Clone this repo into your `~/public_binder/nemos` directory and symlink the
   contents of the `binder/` directory:
   
   ```bash
   git clone git@github.com:flatironinstitute/nemos-workshop-feb-2024.git
   ln -sv nemos-workshop-feb-2024/binder/* .
   ```

5. In your browser, go to `https://binder.flatironinstitute.org` and enter the
   username of whoever create the directory as the Owner and `nemos` as the
   project.
