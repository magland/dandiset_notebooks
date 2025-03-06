Create a jupytext notebook.py with "# %%" delimiters. This will be an introduction to the focus Dandiset with illustrative plots. Please include plenty of comments to describe the experiment and the data. The examples and plots should help the user get started in loading and processing the data. Please be thorough in covering what is available in the dataset. It would also be good to illustrate how the user might use the data to test a scientific hypothesis.

For all generated notebooks, prominently inform the user that the notebook was AI-generated and has not been fully verified, and that they should be cautious when interpreting the code or results.

You should make use of the tools in neurosift-tools to generate the content for this directory.

Assume that all the packages you would need are already installed on the user's system.

You should use `# %% [markdown]` for markdown cells

If you load data from only select files, then you should indicate which files you are using.

Note that it doesn't work to try to index an h5py.Dataset with a numpy array of indices.

Note that you cannot do operations like np.sum over a h5py.Dataset. You need to get a numpy array using something like dataset[:]

If you are going to load a subset of data, it doesn't make sense to load all of the timestamps in memory and then select a subset. Instead, you should load the timestamps for the subset of data you are interested in. So we shouldn't ever see something like `dataset.timestamps[:]` unless we intend to load all the timestamps.

When loading data for illustration, be careful about the size of the data, since the files are hosted remotely and datasets are streamed over the network. You may want to load subsets of data. But if you do, please be sure to indicate that you are doing so, so the reader doesn't get the wrong impression about the data.

If you want to investigate some aspects of the data prior to making the notebook, you can write a temporary python script called something like scratch1.py and run it to see the output. This may give you information about the content of the data that you wouldn't have by just using the available tools. However, it's important that you use the nwb_file_info tool from neurosift-tools prior to trying to do anything with an NWB file.

Keep in mind that through your tool calls you have been given information about what data is available in the files, whereas the reader of the notebook does not have access to that information. So in your illustration it would be helpful to show how they could get that information (e.g., columns in a table, etc).

Do not worry about pynwb typing errors from the linter.

After you have created notebook.py can ask invite the user to try it out.

Once they are satisfied, you can convert to notebook.ipynb using `jupytext --to notebook notebook.py`

Finally, use nbconvert to run that notebook and fill in the output cells in place.