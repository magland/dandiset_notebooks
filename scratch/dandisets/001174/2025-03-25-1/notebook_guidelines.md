The goal is to create a notebook that serves as an introduction to the focus Dandiset with illustrative plots and examples. It will include plenty of comments to describe the experiment and the data. The examples and plots should help the user get started in loading and processing the data. Please be thorough in covering what is available in the dataset. It would also be good to illustrate how the user might use the data to test a scientific hypothesis. In general, the notebook should have plenty of high quality illustrations and plots.

Before creating the notebook, you are going to do some research. You'll need the neurosift-tools and plot-vision MCPs. If those are not installed, prompt the user to install them.

First use neurosift-tools to learn how to load the data. Before write a script to load any NWB data, it's imperative that you first call the approprate nwb_file_info tool.

Next, create and execute python scripts in the dandiset directory (in a tmp_scripts subdirectory). The scripts can generate text output and plots. The plots image files should also go in the tmp_scripts subdirectory.  You should always use the plot-vision MCP to learn about the graphs that you create. This will help you know whether the graphs are informative enough to include in the notebook as well as information about the data that will help you make decisions and know how to describe things in the notebook. Both the script outputs and plots will help inform you about what to put in the notebook. Feel free to run as many scripts as you need to gather the information required to make a good notebook. The more quality information you have, the better you will be able to do in making the notebook. Include comments at the top of each script explaining what information you are trying to obtain with the script.

REMEMBER: After you create each script, be sure to execute it so that you can use the output information.

REMEMBER: After executing each script, if you created plots, please use the plot-vision MCP to review the plots and make sure they are informative and useful. If they are not, you may need to adjust the script and re-run it.

IMPORTANT: Every good quality plot produced by the scripts should be included in the final notebook.

You may see other tmp_scripts directories in this repository. Do not pay attention to those. You should choose the naming of scripts that is appropriate for this Dandiset regardless of what scripts might be named in other places in the repo.

Then create a jupytext notebook.py in the Dandiset directory. Use "# %%" delimiters for the cells.

Prominently inform the user that the notebook was AI-generated with human supervision and has not been fully verified, and that they should be cautious when interpreting the code or results.

Assume that all the packages you would need are already installed on the user's system.

You should use `# %% [markdown]` for markdown cells

If you load data from only select files, then you should indicate which files you are using.

Note that it doesn't work to try to index an h5py.Dataset with a numpy array of indices.

Note that you cannot do operations like np.sum over a h5py.Dataset. You need to get a numpy array using something like dataset[:]

If you are going to load a subset of data, it doesn't make sense to load all the timestamps in memory and then select a subset. Instead, you should load the timestamps for the subset of data you are interested in. So we shouldn't ever see something like `dataset.timestamps[:]` unless we intend to load all the timestamps.

When loading data for illustration, be careful about the size of the data, since the files are hosted remotely and datasets are streamed over the network. You may want to load subsets of data. But if you do, please be sure to indicate that you are doing so, so the reader doesn't get the wrong impression about the data.

Keep in mind that through your tool calls you have been given information about what data are available in the files, whereas the reader of the notebook does not have access to that information. So in your illustration it would be helpful to show how they could get that information (e.g., columns in a table, etc).

When showing unit IDs or channel IDs, be sure to use the actual IDs rather than just the indices.

Do not worry about pynwb typing errors from the linter.

After you have created notebook.py you should run it and make sure it executes without errors. Correct the errors as needed.

You should then convert the notebooks to notebook.ipynb using `jupytext --to notebook notebook.py`

Finally, use nbconvert to run that notebook and fill in the output cells in place.
