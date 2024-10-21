# NFL & CFB Over Under Models

## Overview
This project was designed to analyze Over/Under lines in sportsbooks with the goal of finding "mispriced" lines and taking advantage of them. The process for developing the college model and the NFL model are a little bit different due to available data sources. You'll run the model each week after updating the data, then plug the CSV into R in order to get visually appealing graphics each week. Here's a couple of examples:

![NFL Week 7](nfl_predictions.png)

![CFB Oct 19](my_bets_one.png)

## Setup 
1) Clone the repository

    `git clone https://github.com/prestonf99/CFB-NFL-Over-Under`

2) Install the necessary python packages

    `pip install -r requirements.txt`
    
3) Get an R kernel set up in your jupyter. I personally stumbled through it, so refer to documentation/tutorials/chatGPT for the kernel setup. 

    `install.packages(c("tidyverse", "ggrepel", "ggplot2", "dplyr", "gt", "hms"))`
    
    `if (!requireNamespace("remotes", quietly = TRUE)) {install.packages("remotes")}`
    
    `remotes::install_github("nflverse/nflreadr")`
    
    `remotes::install_github("nflverse/nflplotR")`
    
    `remotes::install_github("nflverse/cfbplotR")`
    
     `remotes::install_github("nflverse/cfbfastR")`

4) Run through the `ASSEMBLE_NFL.ipynb` and `ASSEMBLE_CFB.ipynb` files to get the data prepared for visualizations. For the CFB data, you'll need to retrieve the data from `CFBFastR.ipynb` before assembling it.

5) Use `R_Photos.ipynb` and `CFB_R.ipynb` to load the visualizations.
    - In the CFB assembly, you can choose your criteria for what constitutes a bet (for me, it's > 50% confidence), and make visuals just for those games. 

