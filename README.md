# Earning_Calls

PROBLEM STATEMENT
Earnings Call Transcripts

Project Overview:

Earning Call reports are a valuable data source of insights / signals. Sites like SeekingAlpha provide the transcriptions. These transcriptions can be directly used as inputs for Natural Language Processing (NLP) analyses.

Project Description/Task:

Providing businesses with an intelligent way to extract the data by including following features:

 

1. Summarization of the calls 

The aim is to generate a 1-2 paragraph summary of the salient points of discussion from the call. 

Bonus: Ability to do topic-modelling to extract various topics of discussion in a call. This is important if/when a particular call includes multiple topics of discussion 

2. Extracting / scoring sentiment of the calls 

We want to understand the speaker’s attitude/tone in their statements and classify them under one of positive, negative or neutral. 

3. Extracting / scoring linguistic complexity of the calls 

The intent behind choosing one or more of these techniques is to see if we can derive any insight from how sentences are constructed by speakers on these earning calls. It is said that there exists a positive correlation between earnings management and annual report readability1 and how companies managing their earnings tend to make annual report readability more complex. 

1 Ajina, A., Laouiti, M., & Msolli, B. (2016, July 12). Guiding through the Fog: Does annual report readability reveal earnings management? 

Retrieved from-  https://www.sciencedirect.com/science/article/pii/S0275531916301611 

There are many ways to assess readability and linguistic complexity, some of those techniques being: 

1)The Dale–Chall formula
2)The Gunning fog formula
3)Fry readability graph
4)McLaughlin’s SMOG formula
5)The FORCAST formula
6)Readability and newspaper readership
7)Flesch Scores

4. Entity extraction 

What entities do we want? People, companies, products, dates/timelines, revenue/any numeric data 

We would like to extract factual information from calls and present them in a readily consumable form. Who were the people on the call? What companies? What products were discussed? Any relevant dates/times/revenue/other numeric data we should be aware of? 

This is best structured in a way that can be slotted into a data frame that allows us to utilise this information extracted. Additionally we would also like to see what each person on the call said, allowing us to extract dialogues for a particular speaker from a call. 

5. Clustering/nearest neighbours of trade ideas/companies with similarities in a quarter
Can we find companies/trade ideas that follow similar sentiments or similar themes as determined by previous tasks in a defined time period? This applies to the entire data set and not individual earnings calls. 



Input Data 
Get the Earning call transcripts for google, apple, amazon, Investment banks and of various other companies using web scrapping techniques.
https://seekingalpha.com/article/4341792-apple-inc-aapl-ceo-tim-cook-on-q2-2020-results-earnings-call-transcript
https://seekingalpha.com/symbol/AAPL/earnings/transcripts




