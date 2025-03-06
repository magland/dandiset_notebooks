The goal is to create a notebook that serves as an introduction to the focus Dandiset with illustrative plots and examples. It will include plenty of comments to describe the experiment and the data. The examples and plots should help the user get started in loading and processing the data. Please be thorough in covering what is available in the dataset. It would also be good to illustrate how the user might use the data to test a scientific hypothesis.

Before creating the notebook, you are going to do some research.

First use neurosift-tools to learn how to load the data. Before write a script to load any NWB data, it's imperative that you first call the approprate nwb_file_info tool.

Next, create and execute python scripts in the dandiset directory (in a tmp_scripts subdirectory). The scripts can generate text output and plots. The plots image files should also go in the tmp_scripts subdirectory.  You won't be able to see the plots, so you should ask the user about them to gain insight. These scripts and plots will help inform you about what to put in the notebook. Feel free to make as many as you need to gather the information required to make a good notebook. The more quality information you have, the better you will be able to do in making the notebook.

REMEMBER: After you create each script, be sure to execute it so that you can use the output information.

REMEMBER: After executing each script, if you created plots, please ask the user to comment on the plots before proceeding.

Then create a jupytext notebook.py in the Dandiset directory. Use "# %%" delimiters for the cells.

Prominently inform the user that the notebook was AI-generated with human supervision and has not been fully verified, and that they should be cautious when interpreting the code or results.

Assume that all the packages you would need are already installed on the user's system.

You should use `# %% [markdown]` for markdown cells

If you load data from only select files, then you should indicate which files you are using.

Note that it doesn't work to try to index an h5py.Dataset with a numpy array of indices.

Note that you cannot do operations like np.sum over a h5py.Dataset. You need to get a numpy array using something like dataset[:]

If you are going to load a subset of data, it doesn't make sense to load all of the timestamps in memory and then select a subset. Instead, you should load the timestamps for the subset of data you are interested in. So we shouldn't ever see something like `dataset.timestamps[:]` unless we intend to load all the timestamps.

When loading data for illustration, be careful about the size of the data, since the files are hosted remotely and datasets are streamed over the network. You may want to load subsets of data. But if you do, please be sure to indicate that you are doing so, so the reader doesn't get the wrong impression about the data.

Keep in mind that through your tool calls you have been given information about what data are available in the files, whereas the reader of the notebook does not have access to that information. So in your illustration it would be helpful to show how they could get that information (e.g., columns in a table, etc).

Do not worry about pynwb typing errors from the linter.

After you have created notebook.py can ask invite the user to try it out.

Once they are satisfied, you can convert to notebook.ipynb using `jupytext --to notebook notebook.py`

Finally, use nbconvert to run that notebook and fill in the output cells in place.