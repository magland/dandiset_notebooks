# Dandiset Notebooks

This repository helps create interactive notebooks for exploring and analyzing DANDI datasets using Cline and the neurosift-tools MCP.

## Usage

To create a notebook for a DANDI dataset, tell Cline:

```
Let's make a notebook for Dandiset XXXXXX
```

Replace XXXXXX with the Dandiset ID you want to analyze (e.g., 001341).

Cline will:
1. Create a new timestamped directory in `scratch/dandisets/XXXXXX/yyyy-mm-dd-x`
2. Create a readme.md based on the template
3. Copy the notebook_guidelines.md to the directory
4. Follow the instructions to create a notebook with your help, including:
   - Researching the Dandiset using neurosift-tools
   - Creating and running exploratory Python scripts
   - Generating informative plots for analysis
   - Building an interactive notebook that introduces the dataset
   - Converting and executing the final notebook

## Project Structure

```
.
├── dandisets/          # Completed notebooks
├── scratch/            # Work in progress notebooks
├── templates/          # Template files

