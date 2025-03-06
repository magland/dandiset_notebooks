# Dandiset Notebooks

This repository helps create interactive notebooks for exploring and analyzing DANDI datasets using Cline and the neurosift-tools MCP.

## Usage

First clone this repo and open in it VS Code. You'll need to install and configure the Cline VS Code extension.

To create a notebook for a DANDI dataset, simply tell Cline:

> Let's make a notebook for Dandiset XXXXXX

Replace XXXXXX with the Dandiset ID you want to analyze (e.g., 001341).

Cline will:
1. Create a new timestamped directory in `scratch/dandisets/XXXXXX/yyyy-mm-dd-x`
2. Create a readme.md based on the template
3. Copy the notebook_guidelines.md to the directory
4. Cline will then follow the instructions contained in those files to create a notebook with your help, including:
   - Researching the Dandiset using neurosift-tools
   - Creating and running exploratory Python scripts
   - Generating informative plots for analysis
   - Building an interactive notebook that introduces the dataset
   - Converting and executing the final notebook

Since the Cline cannot view and interpret plots and graphs, you'll need to be the eyes for Cline. At various points, it will ask you to describe what you see in the .png files that it creates. This will help it to know how to make an informative notebook.


## Project Structure

```
.
├── dandisets/          # Completed notebooks
├── scratch/            # Work in progress notebooks
├── templates/          # Template files

